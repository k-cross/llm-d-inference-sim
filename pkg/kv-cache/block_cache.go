/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package kvcache

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/llm-d/llm-d-inference-sim/pkg/retention"
)

const (
	capacityError      = "the kv cache does not have sufficient capacity to store this request"
	delay              = time.Second
	topicNamePrefix    = "kv"
	topicNameSeparator = "@"
)

// Request defines the interface for requests that can be stored in the block cache
// contains sub-set of openai server api request fields that are relevant for the block cache
type Request interface {
	GetRequestID() string
	GetDisplayedModel() string
	GetLoraName() *string
	GetLoraID() *int
	// GetRetentionDirective returns the KV-cache retention directive attached to this
	// request (RFC-0001), or nil when the request is unmarked (plain LRU).
	GetRetentionDirective() *retention.RetentionDirective
}

type blockKey struct {
	hash      uint64
	modelName string
}

// retentionMark is a block's live retention directive (RFC-0001 §1): a numeric priority
// and a wall-clock lease expiry. The expiry is always set to a bounded, non-zero time (see
// applyDirective / retention.EffectiveTTL) -- there are no persistent marks -- and once it
// is in the past the block collapses back to unmarked.
type retentionMark struct {
	priority int
	expiry   time.Time
}

// blockCache represents a thread-safe cache for blocks with eviction policy
type blockCache struct {
	mu              sync.RWMutex
	requestToBlocks map[string][]blockKey              // request id -> array of it blocks (block hashes)
	usedBlocks      map[blockKey]int                   // block hash -> reference count
	unusedBlocks    map[blockKey]time.Time             // block hash -> last usage timestamp
	blockToTokens   map[blockKey][]uint32              // block hash -> block tokens
	retention       map[blockKey]retentionMark         // block hash -> live retention directive (RFC-0001)
	pinnedEvictions int                                // marked-and-unexpired blocks evicted under pressure (RFC-0001 §4)
	loadedModels    map[string]struct{}                // models currently loaded (base model + loaded loras)
	maxBlocks       int                                // maximum number of blocks in the cache
	eventSender     *KVEventSender                     // emits kv events
	eventChan       common.Channel[EventData]          // channel for asynchronous event processing
	usageChan       *common.Channel[common.MetricInfo] // channel for usage reporting
	logger          logr.Logger
	disabled        bool // indicated whether the cache is disabled
}

// newBlockCache creates a new blockCache with the specified maximum number of blocks
func newBlockCache(ctx context.Context, config *common.Configuration, logger logr.Logger,
	usageChan *common.Channel[common.MetricInfo]) (*blockCache, error) {
	if config.IP == "" {
		return nil, errors.New("IP should be defined in the environment (POD_IP)")
	}

	eChan := common.Channel[EventData]{
		Channel: make(chan EventData, 10*config.KVCacheSize),
		Name:    "block cache eventChan",
	}

	var publisher *common.Publisher
	var err error
	if config.ZMQEndpoint != "" {
		publisher, err = common.NewPublisher(ctx, config.ZMQEndpoint)
		if err != nil {
			return nil, err
		}
	}

	eventSender := NewKVEventSender(publisher, CreateKVEventsTopic(config.IP, config.Model),
		eChan, config.EventBatchSize, config.TokenBlockSize, delay, config.UseVllmMapEventFormat, logger)

	bCache := blockCache{
		requestToBlocks: make(map[string][]blockKey),
		usedBlocks:      make(map[blockKey]int),
		unusedBlocks:    make(map[blockKey]time.Time),
		blockToTokens:   make(map[blockKey][]uint32),
		retention:       make(map[blockKey]retentionMark),
		loadedModels:    make(map[string]struct{}),
		maxBlocks:       config.KVCacheSize,
		eventChan:       eChan,
		usageChan:       usageChan,
		eventSender:     eventSender,
		logger:          logger,
	}

	// mark the base model and all it aliases as always loaded,
	// so its blocks will be evicted with lower priority than blocks of unloaded loras
	bCache.setModelLoaded(config.Model)
	for _, modelName := range config.ServedModelNames {
		bCache.setModelLoaded(modelName)
	}

	return &bCache, nil
}

func (bc *blockCache) start(ctx context.Context) {
	bc.logger.V(logging.INFO).Info("Starting KV cache")
	err := bc.eventSender.Run(ctx)
	if err != nil {
		bc.logger.Error(err, "Sender stopped with error")
	}
}

func (bc *blockCache) discard() {
	bc.logger.V(logging.INFO).Info("Discarding KV cache")

	bc.mu.Lock()
	defer bc.mu.Unlock()

	bc.disabled = true

	bc.requestToBlocks = make(map[string][]blockKey)
	bc.usedBlocks = make(map[blockKey]int)
	bc.unusedBlocks = make(map[blockKey]time.Time)
	bc.blockToTokens = make(map[blockKey][]uint32)
	bc.retention = make(map[blockKey]retentionMark)

	common.WriteToChannel(bc.eventChan,
		EventData{action: eventActionAllBlocksCleared},
		bc.logger)
}

func (bc *blockCache) activate() {
	bc.logger.V(logging.INFO).Info("Activating KV cache")

	bc.mu.Lock()
	defer bc.mu.Unlock()

	bc.disabled = false
}

// startRequest adds a request with its associated block hashes to the cache
// and returns the number of blocks that were already in the cache
// model name is the name of the model for the current request, used for eviction policy
func (bc *blockCache) startRequest(req Request, blockHashes []uint64, blockTokens [][]uint32) (int, error) {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	if bc.disabled {
		bc.logger.V(logging.TRACE).Info("KV cache is disabled, request is not added to the kv cache")
		return 0, nil
	}

	if _, exists := bc.requestToBlocks[req.GetRequestID()]; exists {
		// request with the same id already exists
		return 0, fmt.Errorf("request already exists for id %s", req.GetRequestID())
	}

	if len(blockHashes) != len(blockTokens) {
		return 0, fmt.Errorf("invalid input parameters, %d block hashes, %d block tokens", len(blockHashes), len(blockTokens))
	}

	// divide list of blocks to three lists:
	// blockAlreadyInUse - blocks, which are already used by currently running request
	// blockToMoveToUsed - blocks, which were used in the past
	// blocksToAdd - new blocks
	// blocksToAdd holds new blocks paired with their original blockTokens index,
	// so that tokens can be written after the capacity check (avoiding orphaned
	// blockToTokens entries if the check returns an error).
	type newBlock struct {
		key      blockKey
		tokenIdx int
	}
	blocksToAdd := make([]newBlock, 0)
	blockToMoveToUsed := make([]blockKey, 0)
	blockAlreadyInUse := make([]blockKey, 0)

	// first step - ensure that there is enough space for all blocks
	// count number of new blocks + number of blocks that are in the unused blocks
	// don't update the data until we are sure that it's ok

	// lastCachedIdx is the position of the last block in blockHashes that was
	// already in the cache. Used below to set parent_block_hash on the store event.
	// Block hashes form a chained prefix sequence, so cached blocks always appear
	// as a contiguous leading prefix — the parent of the first new block is
	// always blockHashes[lastCachedIdx].
	lastCachedIdx := -1
	for i, blockHash := range blockHashes {
		bKey := blockKey{hash: blockHash, modelName: req.GetDisplayedModel()}
		if _, exists := bc.unusedBlocks[bKey]; exists {
			blockToMoveToUsed = append(blockToMoveToUsed, bKey)
			lastCachedIdx = i
		} else if _, exists := bc.usedBlocks[bKey]; !exists {
			// new block — record its index so tokens can be written after
			// the capacity check passes, preventing orphaned entries on error.
			blocksToAdd = append(blocksToAdd, newBlock{key: bKey, tokenIdx: i})
		} else {
			blockAlreadyInUse = append(blockAlreadyInUse, bKey)
			lastCachedIdx = i
		}
	}

	if len(bc.usedBlocks)+len(blocksToAdd)+len(blockToMoveToUsed) > bc.maxBlocks {
		return 0, errors.New(capacityError)
	}

	// for blocks that are already in use - update the reference
	for _, block := range blockAlreadyInUse {
		bc.usedBlocks[block] += 1
	}

	// for block used in the past - move them to the used blocks collection
	for _, block := range blockToMoveToUsed {
		bc.usedBlocks[block] = 1
		delete(bc.unusedBlocks, block)
	}

	// for new block - add them, if there is no empty slots - evict a block using priority:
	// 1. oldest unused block of an unloaded model
	// 2. oldest unused block of any model
	hashes := []uint64{}
	tokens := []uint32{}

	for _, block := range blocksToAdd {
		if len(bc.usedBlocks)+len(bc.unusedBlocks) == bc.maxBlocks {
			// cache is full but contains unused blocks - evict one block
			evictHash := bc.pickBlockToEvict()
			delete(bc.unusedBlocks, evictHash)
			delete(bc.retention, evictHash)
			common.WriteToChannel(bc.eventChan,
				EventData{action: eventActionRemove, hashes: []uint64{evictHash.hash},
					tokens: bc.blockToTokens[evictHash]},
				bc.logger)
			delete(bc.blockToTokens, evictHash)
		}

		// Commit tokens now that capacity is confirmed.
		bc.blockToTokens[block.key] = blockTokens[block.tokenIdx]

		// Add the new block
		bc.usedBlocks[block.key] = 1

		hashes = append(hashes, block.key.hash)
		tokens = append(tokens, bc.blockToTokens[block.key]...)
	}

	if len(hashes) > 0 {
		// parent is the last already-cached block; nil when all blocks are new.
		var parentHash *uint64
		if lastCachedIdx >= 0 {
			ph := blockHashes[lastCachedIdx]
			parentHash = &ph
		}
		common.WriteToChannel(bc.eventChan,
			EventData{
				action:     eventActionStore,
				hashes:     hashes,
				tokens:     tokens,
				parentHash: parentHash,
				loraName:   req.GetLoraName(),
				loraID:     req.GetLoraID(),
			}, bc.logger)
	}

	// store the request mapping
	// store blockKeys and not only plain uint64
	bc.requestToBlocks[req.GetRequestID()] = make([]blockKey, len(blockHashes))
	for i, blockHash := range blockHashes {
		bKey := blockKey{hash: blockHash, modelName: req.GetDisplayedModel()}
		bc.requestToBlocks[req.GetRequestID()][i] = bKey
	}

	// Apply the request's retention directive to its whole prefix (RFC-0001 §2, router
	// scope). A request with no directive resets its prefix to plain LRU, mirroring the
	// offline sim's pin-clearing (RFC-0001 §1) so a mark can be demoted by later traffic and
	// not only escalated. The clear only runs when some mark actually exists, keeping the
	// common non-agentic workload on the zero-overhead fast path.
	reqBlocks := bc.requestToBlocks[req.GetRequestID()]
	if directive := req.GetRetentionDirective(); directive != nil {
		now := time.Now()
		for _, bKey := range reqBlocks {
			bc.applyDirective(bKey, directive, now)
		}
	} else if len(bc.retention) > 0 {
		for _, bKey := range reqBlocks {
			delete(bc.retention, bKey)
		}
	}

	if bc.usageChan != nil {
		usage := common.MetricInfo{
			Value: float64(len(bc.usedBlocks)) / float64(bc.maxBlocks),
		}
		common.WriteToChannel(*bc.usageChan, usage, bc.logger)
	}
	return len(blockAlreadyInUse) + len(blockToMoveToUsed), nil
}

// finishRequest processes the completion of a request, decreasing reference counts
func (bc *blockCache) finishRequest(requestID string) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	if bc.disabled {
		bc.logger.V(logging.TRACE).Info("KV cache is disabled, request completion is not processed by the kv cache")
		return nil
	}

	// Get blocks associated with this request
	blockHashes, exists := bc.requestToBlocks[requestID]
	if !exists {
		return nil
	}

	now := time.Now()

	// Decrease reference count for each block
	errBlocks := make([]blockKey, 0)
	for _, blockHash := range blockHashes {
		if refCount, exists := bc.usedBlocks[blockHash]; exists {
			if refCount > 1 {
				// this block is in use by another request, just update reference count
				bc.usedBlocks[blockHash] = refCount - 1
			} else {
				// this was the last block usage - move this block to unused
				bc.unusedBlocks[blockHash] = now
				delete(bc.usedBlocks, blockHash)
			}
		} else {
			errBlocks = append(errBlocks, blockHash)
		}
	}

	if bc.usageChan != nil {
		usage := common.MetricInfo{
			Value: float64(len(bc.usedBlocks)) / float64(bc.maxBlocks),
		}
		common.WriteToChannel(*bc.usageChan, usage, bc.logger)
	}

	// Remove the request mapping
	delete(bc.requestToBlocks, requestID)

	if len(errBlocks) > 0 {
		var builder strings.Builder

		for i, b := range errBlocks {
			if i > 0 {
				builder.WriteString(", ")
			}
			fmt.Fprintf(&builder, "%d(%s)", b.hash, b.modelName)
		}
		return fmt.Errorf("not existing blocks %s for request %s", builder.String(), requestID)
	}

	return nil
}

// GetStats returns current cache statistics (for testing/debugging)
func (bc *blockCache) getStats() (int, int, int) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	return len(bc.requestToBlocks), len(bc.usedBlocks) + len(bc.unusedBlocks), len(bc.unusedBlocks)
}

// getBlockInfo returns reference count and if it's in the cache for a specific block (for testing)
// if block is in use by currently running requests the count will be positive, boolean is true
// if block is in the unused list - count is 0, boolean is true
// if block is not in both collections - count is 0, boolean is false
func (bc *blockCache) getBlockInfo(blockHash blockKey) (int, bool) {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	refCount, exists := bc.usedBlocks[blockHash]
	if exists {
		return refCount, true
	}
	_, exists = bc.unusedBlocks[blockHash]
	if exists {
		return 0, true
	}

	return 0, false
}

// countCachedBlockPrefix returns the number of continuous blocks from the given list that are already in the cache
func (bc *blockCache) countCachedBlockPrefix(blockHashes []uint64, modelName string) int {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	if bc.disabled {
		return 0
	}

	var count int
	for _, blockHash := range blockHashes {
		bKey := blockKey{hash: blockHash, modelName: modelName}
		// Check if block is in used blocks (currently in use by running requests)
		if _, exists := bc.usedBlocks[bKey]; exists {
			count++
		} else if _, exists := bc.unusedBlocks[bKey]; exists {
			// Check if block is in unused blocks (was used in past)
			count++
		} else {
			// return count once a block is not found in the cache
			return count
		}
	}
	return count
}

// lruCandidate tracks the most-evictable block within an LRU-ordered bucket: an
// unloaded-model block outranks a loaded-model one, and within the same class the oldest
// (least-recently-used) wins -- the base eviction order upstream uses.
type lruCandidate struct {
	key    blockKey
	loaded bool
	t      time.Time
	have   bool
}

// consider replaces the incumbent when bk is a stronger eviction target than it.
func (c *lruCandidate) consider(bk blockKey, loaded bool, t time.Time) {
	if c.have {
		if loaded != c.loaded {
			if loaded {
				return // an unloaded incumbent always beats a loaded candidate
			}
		} else if !t.Before(c.t) {
			return // same class, incumbent is at least as old (more evictable)
		}
	}
	c.key, c.loaded, c.t, c.have = bk, loaded, t, true
}

// pickBlockToEvict selects the unused block to evict, honoring the RFC-0001 §3 retention
// ranks layered on the base unloaded-model-first LRU. Effective rank orders
// evict-first < unmarked < marked (priority ascending); TTL expiry collapses a mark to
// unmarked. The lowest non-empty rank is drained first, so a marked block is only
// sacrificed when nothing cheaper is resident -- and that sacrifice is counted as pinned
// pressure (RFC-0001 §4). A single pass buckets every unused block by rank and prunes
// expired leases as it goes, so the no-directive fast path re-engages once all leases
// lapse. Must be called with bc.mu held.
func (bc *blockCache) pickBlockToEvict() blockKey {
	// Fast path: no live directives anywhere -> plain unloaded-first LRU, no per-block
	// retention lookups, preserving zero overhead for non-agentic workloads.
	if len(bc.retention) == 0 {
		var lru lruCandidate
		for bk, t := range bc.unusedBlocks {
			_, loaded := bc.loadedModels[bk.modelName]
			lru.consider(bk, loaded, t)
		}
		return lru.key
	}

	now := time.Now()
	var evictFirst, unmarked lruCandidate
	var marked blockKey
	var markedMark retentionMark
	var markedTime time.Time
	haveMarked := false

	for bk, t := range bc.unusedBlocks {
		mark, isMarked := bc.retention[bk]
		if isMarked && !now.Before(mark.expiry) {
			delete(bc.retention, bk) // lease lapsed -> prune and treat as unmarked
			isMarked = false
		}
		_, loaded := bc.loadedModels[bk.modelName]
		switch {
		case !isMarked:
			unmarked.consider(bk, loaded, t)
		case mark.priority < 0:
			evictFirst.consider(bk, loaded, t)
		case !haveMarked || markLessValuable(mark.priority, mark.expiry, t,
			markedMark.priority, markedMark.expiry, markedTime):
			marked, markedMark, markedTime, haveMarked = bk, mark, t, true
		}
	}

	if evictFirst.have {
		return evictFirst.key
	}
	if unmarked.have {
		return unmarked.key
	}
	// Every unused block is marked-and-unexpired: sacrifice the least valuable pin and
	// record the pressure (RFC-0001 §1/§4).
	bc.pinnedEvictions++
	return marked
}

// markLessValuable reports whether mark A is a better eviction target than mark B: lower
// priority wins; on a tie the soonest-expiring lease; on a further tie the older block.
// (All marks carry a bounded, non-zero expiry, so there is no persistent-lease case.)
func markLessValuable(prioA int, expiryA, timeA time.Time, prioB int, expiryB, timeB time.Time) bool {
	if prioA != prioB {
		return prioA < prioB
	}
	if !expiryA.Equal(expiryB) {
		return expiryA.Before(expiryB)
	}
	return timeA.Before(timeB)
}

// getPinnedEvictions returns the number of marked-and-unexpired blocks evicted under
// capacity pressure -- the router's over-pin signal (RFC-0001 §4).
func (bc *blockCache) getPinnedEvictions() int {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	return bc.pinnedEvictions
}

// applyDirective records (or escalates) a block's retention mark from a request's directive.
// A shared block carries the maximum of the live directives covering it (RFC-0001 §1): a
// higher priority wins, and on a tie the later-expiring lease is kept. The lease is bounded
// by the server policy (retention.EffectiveTTL), so the mark is never persistent. Priority
// is trusted to be in range -- ParseRetentionHeader has already rejected out-of-range hints.
func (bc *blockCache) applyDirective(bk blockKey, directive *retention.RetentionDirective, now time.Time) {
	newPriority := directive.Priority
	newExpiry := now.Add(retention.EffectiveTTL(directive.TTL))

	if existing, ok := bc.retention[bk]; ok {
		if now.Before(existing.expiry) &&
			!markLessValuable(existing.priority, existing.expiry, now, newPriority, newExpiry, now) {
			return // existing mark is still live and at least as strong -> keep it
		}
	}
	bc.retention[bk] = retentionMark{priority: newPriority, expiry: newExpiry}
}

func (bc *blockCache) setModelLoaded(model string) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	bc.loadedModels[model] = struct{}{}
}

func (bc *blockCache) setModelUnloaded(model string) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	delete(bc.loadedModels, model)
}

// ZMQ topic format is: kv@<pod-ip>@<model-name>
func CreateKVEventsTopic(ip string, model string) string {
	return topicNamePrefix + topicNameSeparator + ip + topicNameSeparator + model
}
