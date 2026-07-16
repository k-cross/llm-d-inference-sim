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
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/retention"
)

// These plain-Go tests exercise the RFC-0001 priority evictor directly against the
// blockCache, mirroring the offline sim's executable spec (bench/sim/node.py). A block is
// only an eviction candidate once its request has finished (moved to unusedBlocks), so each
// helper starts *and* finishes a request to make its block evictable, then a final
// startRequest with no finish drives the eviction under test.

const evTestModel = "model"

func newTestBlockCache(t *testing.T, capacity int) *blockCache {
	t.Helper()
	config := &common.Configuration{IP: localhost, Port: 1234, Model: evTestModel, KVCacheSize: capacity}
	bc, err := newBlockCache(context.Background(), config, logr.Discard(), nil)
	if err != nil {
		t.Fatalf("newBlockCache: %v", err)
	}
	return bc
}

// storeFinished admits a single-block request carrying directive d and immediately finishes
// it, leaving the block resident-but-unused (an eviction candidate) with its mark applied.
func storeFinished(t *testing.T, bc *blockCache, id string, hash uint64, d *retention.RetentionDirective) {
	t.Helper()
	req := &testRequest{
		id:          id,
		model:       evTestModel,
		blockHashes: []uint64{hash},
		tokens:      [][]uint32{{uint32(hash)}},
		directive:   d,
	}
	if _, err := bc.startRequest(req, req.blockHashes, req.tokens); err != nil {
		t.Fatalf("startRequest %s: %v", id, err)
	}
	if err := bc.finishRequest(id); err != nil {
		t.Fatalf("finishRequest %s: %v", id, err)
	}
}

// triggerEviction admits a fresh unmarked block (leaving it in use) to force one eviction
// among the currently-unused blocks.
func triggerEviction(t *testing.T, bc *blockCache, id string, hash uint64) {
	t.Helper()
	req := &testRequest{
		id:          id,
		model:       evTestModel,
		blockHashes: []uint64{hash},
		tokens:      [][]uint32{{uint32(hash)}},
	}
	if _, err := bc.startRequest(req, req.blockHashes, req.tokens); err != nil {
		t.Fatalf("trigger startRequest %s: %v", id, err)
	}
}

func resident(bc *blockCache, hash uint64) bool {
	_, ok := bc.getBlockInfo(blockKey{hash: hash, modelName: evTestModel})
	return ok
}

func directive(priority int, ttl time.Duration) *retention.RetentionDirective {
	return &retention.RetentionDirective{Priority: priority, TTL: ttl}
}

// evict-first is below-normal: it is dropped before plain unmarked traffic, even when the
// unmarked block is older (what LRU alone would evict first).
func TestEvictFirstDroppedBeforeUnmarked(t *testing.T) {
	bc := newTestBlockCache(t, 2)
	storeFinished(t, bc, "b", 2, nil)                                        // unmarked, older
	storeFinished(t, bc, "a", 1, directive(retention.EvictFirstPriority, 0)) // evict-first, newer
	triggerEviction(t, bc, "c", 3)

	if resident(bc, 1) {
		t.Error("evict-first block 1 should have been dropped first")
	}
	if !resident(bc, 2) {
		t.Error("unmarked block 2 should have survived despite being older")
	}
	if got := bc.getPinnedEvictions(); got != 0 {
		t.Errorf("pinnedEvictions = %d, want 0 (evicting evict-first is not pinned pressure)", got)
	}
}

// rank order evict-first < unmarked < marked: a HIGH block outlives an evict-first one.
func TestMarkedSurvivesEvictFirst(t *testing.T) {
	bc := newTestBlockCache(t, 2)
	storeFinished(t, bc, "a", 1, directive(retention.HighPriority, time.Hour))
	storeFinished(t, bc, "b", 2, directive(retention.EvictFirstPriority, 0))
	triggerEviction(t, bc, "c", 3)

	if !resident(bc, 1) {
		t.Error("HIGH block 1 should have survived")
	}
	if resident(bc, 2) {
		t.Error("evict-first block 2 should have been dropped")
	}
}

// all resident blocks marked-and-unexpired: sacrifice the lowest-priority one and record
// the pinned pressure (RFC-0001 §1/§4).
func TestLowerPriorityMarkedEvictedUnderPressure(t *testing.T) {
	bc := newTestBlockCache(t, 2)
	storeFinished(t, bc, "a", 1, directive(retention.PinnedPriority, time.Hour))
	storeFinished(t, bc, "b", 2, directive(retention.HighPriority, time.Hour))
	triggerEviction(t, bc, "c", 3)

	if !resident(bc, 1) {
		t.Error("PINNED block 1 should have survived")
	}
	if resident(bc, 2) {
		t.Error("lower-priority HIGH block 2 should have been sacrificed")
	}
	if got := bc.getPinnedEvictions(); got != 1 {
		t.Errorf("pinnedEvictions = %d, want 1", got)
	}
}

// TTL expiry collapses a mark to unmarked: a HIGH block whose lease lapsed evicts by plain
// LRU (oldest first) rather than out-surviving unmarked traffic.
func TestExpiredMarkCollapsesToUnmarked(t *testing.T) {
	bc := newTestBlockCache(t, 2)
	storeFinished(t, bc, "a", 1, directive(retention.HighPriority, 20*time.Millisecond)) // older, short lease
	storeFinished(t, bc, "b", 2, nil)                                                    // newer, unmarked
	time.Sleep(50 * time.Millisecond)                                                    // block 1's lease expires
	triggerEviction(t, bc, "c", 3)

	if resident(bc, 1) {
		t.Error("expired HIGH block 1 should collapse to unmarked and be evicted as LRU-oldest")
	}
	if !resident(bc, 2) {
		t.Error("block 2 should have survived")
	}
	if got := bc.getPinnedEvictions(); got != 0 {
		t.Errorf("pinnedEvictions = %d, want 0 (an expired mark is not a pin)", got)
	}
}

// A later unmarked request touching a marked block resets it to plain LRU (RFC-0001 §1),
// so a mark can be demoted by traffic, not only escalated. Without the reset, block 1 would
// stay HIGH and survive; with it, block 1 is just the LRU-oldest unmarked block and goes.
func TestUnmarkedReadmitClearsMark(t *testing.T) {
	bc := newTestBlockCache(t, 3)
	storeFinished(t, bc, "a", 1, directive(retention.HighPriority, time.Minute)) // block 1 HIGH
	storeFinished(t, bc, "a-again", 1, nil)                                      // unmarked re-touch -> clears the mark
	storeFinished(t, bc, "b", 2, nil)                                            // unmarked
	storeFinished(t, bc, "c", 3, nil)                                            // unmarked
	triggerEviction(t, bc, "d", 4)                                               // cache full (3) -> evict one

	if resident(bc, 1) {
		t.Error("block 1's HIGH mark should have been cleared by the unmarked re-admit, evicting it as LRU-oldest")
	}
	if !resident(bc, 2) || !resident(bc, 3) {
		t.Error("blocks 2 and 3 should have survived")
	}
	if got := bc.getPinnedEvictions(); got != 0 {
		t.Errorf("pinnedEvictions = %d, want 0 (nothing is marked after the reset)", got)
	}
}

// An unmarked-only workload never touches retention state: eviction is plain LRU (oldest
// first) and records no pinned pressure -- the zero-overhead path for non-agentic traffic.
func TestUnmarkedWorkloadIsPlainLRU(t *testing.T) {
	bc := newTestBlockCache(t, 2)
	storeFinished(t, bc, "a", 1, nil) // older
	storeFinished(t, bc, "b", 2, nil) // newer
	triggerEviction(t, bc, "c", 3)

	if resident(bc, 1) {
		t.Error("LRU-oldest block 1 should have been evicted")
	}
	if !resident(bc, 2) {
		t.Error("block 2 should have survived")
	}
	if got := bc.getPinnedEvictions(); got != 0 {
		t.Errorf("pinnedEvictions = %d, want 0", got)
	}
}
