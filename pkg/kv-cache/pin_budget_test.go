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

// These tests exercise the E5 pin-budget admission cap (RFC-0001 §4): once the resident
// high+pinned band reaches pin-budget-frac x kv-cache-size, a further pin is degraded to
// plain LRU instead of honored, bounding the over-pinning collapse.

func newBudgetBlockCache(t *testing.T, capacity int, budgetFrac float64) *blockCache {
	t.Helper()
	config := &common.Configuration{
		IP: localhost, Port: 1234, Model: evTestModel,
		KVCacheSize: capacity, PinBudgetFrac: budgetFrac,
	}
	bc, err := newBlockCache(context.Background(), config, logr.Discard(), nil, nil)
	if err != nil {
		t.Fatalf("newBlockCache: %v", err)
	}
	return bc
}

func livePinned(bc *blockCache) int {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	return bc.countPinnedBlocks(time.Now())
}

// A low budget plateaus the live pinned band at pinBudgetBlocks: pins arriving after the cap
// is reached are degraded to plain LRU (left resident but unmarked) and counted.
func TestPinBudgetPlateausAndDegrades(t *testing.T) {
	// capacity 10, frac 0.3 -> budget 3 blocks. Cache is large enough that none of the 5
	// single-block requests evict, so this isolates admission from eviction.
	bc := newBudgetBlockCache(t, 10, 0.3)

	for i := uint64(1); i <= 5; i++ {
		storeFinished(t, bc, "pin", i, directive(retention.HighPriority, time.Hour))
	}

	if got := livePinned(bc); got != 3 {
		t.Errorf("live pinned blocks = %d, want 3 (plateau at the budget)", got)
	}
	if got := bc.budgetDegradations; got != 2 {
		t.Errorf("budgetDegradations = %d, want 2 (pins 4 and 5 refused)", got)
	}
	// The two degraded blocks are resident but carry no retention mark (plain LRU).
	for _, h := range []uint64{4, 5} {
		if !resident(bc, h) {
			t.Errorf("degraded block %d should still be resident (as plain LRU)", h)
		}
		if _, marked := bc.retention[blockKey{hash: h, modelName: evTestModel}]; marked {
			t.Errorf("degraded block %d should not carry a retention mark", h)
		}
	}
}

// The budget is a fit check on the blocks a pin would newly add: a single request whose
// prefix exceeds the whole budget is degraded entirely (no partial pin), rather than
// overshooting the cap by a full prefix the way a coarse "already at cap" rule would.
func TestPinBudgetFitCheckDegradesOversizedPrefix(t *testing.T) {
	bc := newBudgetBlockCache(t, 20, 0.15) // budget = int(0.15*20) = 3 blocks
	req := &testRequest{
		id: "big", model: evTestModel,
		blockHashes: []uint64{1, 2, 3, 4, 5},
		tokens:      [][]uint32{{1}, {2}, {3}, {4}, {5}},
		directive:   directive(retention.HighPriority, time.Hour),
	}
	if _, err := bc.startRequest(req, req.blockHashes, req.tokens); err != nil {
		t.Fatalf("startRequest: %v", err)
	}
	if got := livePinned(bc); got != 0 {
		t.Errorf("live pinned = %d, want 0 (a 5-block prefix does not fit a 3-block budget)", got)
	}
	if got := bc.budgetDegradations; got != 1 {
		t.Errorf("budgetDegradations = %d, want 1", got)
	}
}

// A prefix that fits exactly is admitted whole -- the fit check is inclusive at the budget.
func TestPinBudgetFitCheckAdmitsWhenItFits(t *testing.T) {
	bc := newBudgetBlockCache(t, 20, 0.15) // budget 3 blocks
	req := &testRequest{
		id: "fits", model: evTestModel,
		blockHashes: []uint64{1, 2, 3},
		tokens:      [][]uint32{{1}, {2}, {3}},
		directive:   directive(retention.HighPriority, time.Hour),
	}
	if _, err := bc.startRequest(req, req.blockHashes, req.tokens); err != nil {
		t.Fatalf("startRequest: %v", err)
	}
	if got := livePinned(bc); got != 3 {
		t.Errorf("live pinned = %d, want 3 (a 3-block prefix fits a 3-block budget exactly)", got)
	}
	if got := bc.budgetDegradations; got != 0 {
		t.Errorf("budgetDegradations = %d, want 0", got)
	}
}

// At frac 1.0 (the default, uncapped) the budget is disabled: every pin is honored and no
// degradation is recorded -- the pre-E5 behavior is unchanged.
func TestPinBudgetUncappedAtFracOne(t *testing.T) {
	bc := newBudgetBlockCache(t, 10, 1.0)
	if bc.pinBudgetEnabled {
		t.Fatal("pin budget should be disabled at frac 1.0")
	}
	for i := uint64(1); i <= 5; i++ {
		storeFinished(t, bc, "pin", i, directive(retention.HighPriority, time.Hour))
	}
	if got := livePinned(bc); got != 5 {
		t.Errorf("live pinned blocks = %d, want 5 (uncapped)", got)
	}
	if got := bc.budgetDegradations; got != 0 {
		t.Errorf("budgetDegradations = %d, want 0 (uncapped)", got)
	}
}

// The zero value from a bare Configuration literal (frac 0.0) means unset -> disabled, so
// existing direct-construction tests and non-E5 code paths are unaffected.
func TestPinBudgetDisabledWhenUnset(t *testing.T) {
	bc := newBudgetBlockCache(t, 10, 0.0)
	if bc.pinBudgetEnabled {
		t.Fatal("pin budget should be disabled at the 0.0 zero value")
	}
	for i := uint64(1); i <= 3; i++ {
		storeFinished(t, bc, "pin", i, directive(retention.HighPriority, time.Hour))
	}
	if got := bc.budgetDegradations; got != 0 {
		t.Errorf("budgetDegradations = %d, want 0 (disabled)", got)
	}
}

// Evict-first (below-normal) directives are never budgeted -- they only free cache, so even a
// tiny budget never degrades or counts them.
func TestPinBudgetIgnoresEvictFirst(t *testing.T) {
	bc := newBudgetBlockCache(t, 10, 0.1) // budget 1 block
	for i := uint64(1); i <= 4; i++ {
		storeFinished(t, bc, "ef", i, directive(retention.EvictFirstPriority, time.Hour))
	}
	if got := bc.budgetDegradations; got != 0 {
		t.Errorf("budgetDegradations = %d, want 0 (evict-first is not a pin)", got)
	}
	if got := livePinned(bc); got != 0 {
		t.Errorf("live pinned blocks = %d, want 0 (evict-first is below-normal)", got)
	}
}
