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

// KV-cache retention directives (RFC-0001). A directive is a hint carried from the
// router/framework to the engine asking it to retain (or shed) a request's prefix
// blocks harder or softer than plain LRU. On the wire, priorities are strictly numeric:
// the 0-100 scale plus the -1 evict-first extension. A directive never affects
// correctness -- a malformed one is ignored, and a pin is soft under capacity pressure.
//
// This is a leaf package (stdlib only) so both the request API and the KV-cache evictor can
// import it without an import cycle.
package retention

import (
	"strconv"
	"strings"
	"time"
)

// Numeric priority levels (RFC-0001 §1). The absence of a directive is "normal"/unmarked
// (plain LRU) and is represented by the lack of a directive, never a stored value.
const (
	// EvictFirstPriority marks a block below-normal: a bucket drained before the LRU free
	// list. Expresses "worth less than unmarked traffic", which a retain-only evictor cannot.
	EvictFirstPriority = -1
	// HighPriority is the router's learned-retention level (TTL optional).
	HighPriority = 50
	// PinnedPriority retains hardest (TTL mandatory in a real deployment, server-capped).
	PinnedPriority = 100

	MinPriority = EvictFirstPriority
	MaxPriority = PinnedPriority
)

// Lease policy (RFC-0001 §1). Every mark is a TTL'd lease, never an unbounded pin: a
// directive that carries no lease gets DefaultTTL, and any lease is capped at MaxTTL. This
// is the server-side answer to the OOM/starvation objection (vllm#23083) -- persistence is
// a server policy, not a client right -- so a marked block always eventually reverts to
// plain LRU even if the router never sends an unpin.
const (
	DefaultTTL = 30 * time.Second
	MaxTTL     = 5 * time.Minute
)

// KVCachePriorityHeader is the compact router-path transport (RFC-0001 §2):
//
//	x-kv-cache-priority: <int>[; ttl=<Go duration>][; scope=<id>]
//
// A flat key=value grammar so the extProc and the engine both parse it without JSON.
const KVCachePriorityHeader = "x-kv-cache-priority"

// RetentionDirective is a parsed retention hint scoped to a request's whole prefix.
type RetentionDirective struct {
	// Priority is the numeric rank in [MinPriority, MaxPriority].
	Priority int
	// TTL is the lease duration from receipt; zero means the server applies DefaultTTL.
	TTL time.Duration
	// Scope is the session/program identity the directive belongs to (may be empty).
	Scope string
}

// EffectiveTTL applies the server lease policy to a directive's TTL (RFC-0001 §1): a
// directive with no lease gets DefaultTTL, and every lease is capped at MaxTTL. The result
// is always > 0, so no mark is ever unbounded -- a pinned block still reverts to plain LRU
// once its (capped) lease lapses.
func EffectiveTTL(ttl time.Duration) time.Duration {
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	if ttl > MaxTTL {
		ttl = MaxTTL
	}
	return ttl
}

// ParseRetentionHeader parses an x-kv-cache-priority header value. It returns the parsed
// directive and true on success, or (nil, false) when the header is empty or malformed
// (non-numeric or out-of-range priority) -- the caller ignores and logs a bad hint rather
// than failing the request (RFC-0001 §1). Clamping an out-of-range priority is deliberately
// NOT done: silently rounding e.g. -50 to evict-first would demote a request's prefix below
// normal traffic on a typo. The grammar is a leading integer priority in [MinPriority,
// MaxPriority] followed by optional "; key=value" pairs (ttl, scope); unknown keys are
// ignored so the grammar can grow without breaking older parsers.
func ParseRetentionHeader(value string) (*RetentionDirective, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, false
	}

	parts := strings.Split(value, ";")
	priority, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return nil, false
	}
	if priority < MinPriority || priority > MaxPriority {
		return nil, false // out of range -> malformed -> ignored, not clamped
	}

	d := &RetentionDirective{Priority: priority}
	for _, part := range parts[1:] {
		key, val, found := strings.Cut(part, "=")
		if !found {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "ttl":
			if ttl, err := time.ParseDuration(val); err == nil && ttl > 0 {
				d.TTL = ttl
			}
		case "scope":
			d.Scope = val
		}
	}
	return d, true
}
