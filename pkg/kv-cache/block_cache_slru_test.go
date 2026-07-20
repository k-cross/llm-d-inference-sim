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

// Plain-Go tests for the SLRU eviction policy (E2), in the style of
// retention_eviction_test.go: storeFinished leaves a block resident-but-unused, a final
// startRequest with no finish drives the eviction under test.

func newSLRUBlockCache(t *testing.T, capacity int, ratio float64, mode string) *blockCache {
	t.Helper()
	config := &common.Configuration{
		IP: localhost, Port: 1234, Model: evTestModel, KVCacheSize: capacity,
		EvictionPolicy:         common.EvictionPolicySLRU,
		SLRUProtectedRatio:     ratio,
		RetentionDirectiveMode: mode,
	}
	bc, err := newBlockCache(context.Background(), config, logr.Discard(), nil, nil)
	if err != nil {
		t.Fatalf("newBlockCache: %v", err)
	}
	return bc
}

func isProtected(bc *blockCache, hash uint64) bool {
	_, ok := bc.protected[blockKey{hash: hash, modelName: evTestModel}]
	return ok
}

// A new block enters probation; a second access promotes it to protected; eviction
// drains probation before protected even when the protected block is older.
func TestSLRUSecondAccessPromotesAndProbationDrainsFirst(t *testing.T) {
	bc := newSLRUBlockCache(t, 2, 0.5, common.RetentionModePin)

	storeFinished(t, bc, "a1", 1, nil) // first access -> probation
	if isProtected(bc, 1) {
		t.Fatal("block 1 should enter probation on first access")
	}
	storeFinished(t, bc, "a2", 1, nil) // second access -> protected
	if !isProtected(bc, 1) {
		t.Fatal("block 1 should be protected after a second access")
	}

	storeFinished(t, bc, "b", 2, nil) // probation, newer than block 1
	triggerEviction(t, bc, "c", 3)

	if resident(bc, 1) != true {
		t.Error("protected block 1 should have survived")
	}
	if resident(bc, 2) {
		t.Error("probation block 2 should have been evicted first despite being newer")
	}
	if bc.probationEvictions != 1 || bc.protectedEvictions != 0 {
		t.Errorf("eviction composition = %d/%d, want 1 probation / 0 protected",
			bc.probationEvictions, bc.protectedEvictions)
	}
}

// With no probation candidates the protected LRU is evicted, feeds the ghost set, and a
// later re-insert of the same block skips probation (ghost hit -> protected).
func TestSLRUProtectedEvictionFeedsGhostAndReentryIsProtected(t *testing.T) {
	bc := newSLRUBlockCache(t, 2, 0.5, common.RetentionModePin)

	storeFinished(t, bc, "a1", 1, nil)
	storeFinished(t, bc, "a2", 1, nil) // block 1 protected
	triggerEviction(t, bc, "b", 2)     // block 2 stays in use; only block 1 is evictable

	// Cache is now full (1 unused-protected + 2 in-use... capacity 2: block 1 unused,
	// block 2 in use). The next new block must sacrifice protected block 1.
	triggerEviction(t, bc, "c", 3)
	if resident(bc, 1) {
		t.Fatal("protected block 1 should have been evicted (no probation candidates)")
	}
	if bc.protectedEvictions != 1 {
		t.Fatalf("protectedEvictions = %d, want 1", bc.protectedEvictions)
	}
	if _, ok := bc.ghost[blockKey{hash: 1, modelName: evTestModel}]; !ok {
		t.Fatal("evicted protected block 1 should be in the ghost set")
	}

	// Free capacity, then re-insert block 1: it must skip probation via the ghost set.
	if err := bc.finishRequest("b"); err != nil {
		t.Fatalf("finishRequest b: %v", err)
	}
	if err := bc.finishRequest("c"); err != nil {
		t.Fatalf("finishRequest c: %v", err)
	}
	storeFinished(t, bc, "a3", 1, nil)
	if !isProtected(bc, 1) {
		t.Error("re-inserted block 1 should re-enter protected via the ghost set")
	}
	if bc.ghostHits != 1 {
		t.Errorf("ghostHits = %d, want 1", bc.ghostHits)
	}
	if _, ok := bc.ghost[blockKey{hash: 1, modelName: evTestModel}]; ok {
		t.Error("ghost entry should be consumed by the hit")
	}
}

// Promotion past the protected cap demotes the least-recently-used unused protected
// block back to probation instead of overflowing the segment.
func TestSLRUProtectedCapDemotesLRU(t *testing.T) {
	bc := newSLRUBlockCache(t, 4, 0.5, common.RetentionModePin) // protectedCap = 2

	for _, h := range []uint64{1, 2, 3} {
		storeFinished(t, bc, "first", h, nil)  // first access -> probation
		storeFinished(t, bc, "second", h, nil) // second access -> protected
		time.Sleep(2 * time.Millisecond)       // order unused timestamps deterministically
	}

	if isProtected(bc, 1) {
		t.Error("block 1 (protected LRU) should have been demoted to probation at cap")
	}
	if !isProtected(bc, 2) || !isProtected(bc, 3) {
		t.Error("blocks 2 and 3 should remain protected")
	}
	if !resident(bc, 1) {
		t.Error("demotion must not evict block 1")
	}
}

// In promote directive mode a directive is a segment hint: priority > 0 promotes the
// prefix to protected with no retention mark, and evict-first demotes it back.
func TestSLRUPromoteModeDirectiveIsSegmentHintNotPin(t *testing.T) {
	bc := newSLRUBlockCache(t, 4, 0.5, common.RetentionModePromote)

	storeFinished(t, bc, "a", 1, directive(retention.HighPriority, time.Hour))
	if !isProtected(bc, 1) {
		t.Error("priority>0 directive should promote block 1 to protected on first insert")
	}
	if len(bc.retention) != 0 {
		t.Errorf("promote mode must not record retention marks, got %d", len(bc.retention))
	}

	storeFinished(t, bc, "b", 1, directive(retention.EvictFirstPriority, 0))
	if isProtected(bc, 1) {
		t.Error("evict-first directive should demote block 1 to probation")
	}
	if len(bc.retention) != 0 {
		t.Errorf("promote mode must not record retention marks, got %d", len(bc.retention))
	}
}

// slru + pin mode composes: a live mark still outranks both unmarked segments, and the
// eviction order within unmarked is probation < protected.
func TestSLRUPinModeMarkOutranksSegments(t *testing.T) {
	bc := newSLRUBlockCache(t, 3, 0.9, common.RetentionModePin) // cap 2: room for both

	storeFinished(t, bc, "m", 1, directive(retention.HighPriority, time.Hour)) // marked
	storeFinished(t, bc, "p1", 2, nil)
	storeFinished(t, bc, "p2", 2, nil) // block 2 protected
	triggerEviction(t, bc, "t1", 3)    // fills the cache, no eviction yet
	triggerEviction(t, bc, "t2", 4)    // forces one eviction among {1 marked, 2 protected}

	// Protected-but-unmarked is cheaper than marked, so block 2 goes first.
	if !resident(bc, 1) {
		t.Error("marked block 1 should outrank the protected segment")
	}
	if resident(bc, 2) {
		t.Error("protected-but-unmarked block 2 should be evicted before the marked block")
	}
	if bc.protectedEvictions != 1 {
		t.Errorf("protectedEvictions = %d, want 1", bc.protectedEvictions)
	}
}
