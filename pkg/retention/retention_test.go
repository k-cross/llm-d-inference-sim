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

package retention

import (
	"testing"
	"time"
)

func TestParseRetentionHeader(t *testing.T) {
	cases := []struct {
		name     string
		value    string
		wantOK   bool
		priority int
		ttl      time.Duration
		scope    string
	}{
		{name: "priority only", value: "100", wantOK: true, priority: 100},
		{name: "full directive", value: "50; ttl=3s; scope=sess-7", wantOK: true, priority: 50, ttl: 3 * time.Second, scope: "sess-7"},
		{name: "evict-first", value: "-1", wantOK: true, priority: -1},
		{name: "boundary max", value: "100", wantOK: true, priority: MaxPriority},
		{name: "whitespace tolerant", value: "  50 ; ttl=250ms ", wantOK: true, priority: 50, ttl: 250 * time.Millisecond},
		{name: "priority above range ignored", value: "999", wantOK: false},
		{name: "priority below range ignored", value: "-50", wantOK: false},
		{name: "bad ttl ignored", value: "50; ttl=nonsense", wantOK: true, priority: 50},
		{name: "empty", value: "", wantOK: false},
		{name: "non-numeric priority", value: "high", wantOK: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d, ok := ParseRetentionHeader(tc.value)
			if ok != tc.wantOK {
				t.Fatalf("ParseRetentionHeader(%q) ok = %v, want %v", tc.value, ok, tc.wantOK)
			}
			if !tc.wantOK {
				if d != nil {
					t.Fatalf("expected nil directive on failure, got %+v", d)
				}
				return
			}
			if d.Priority != tc.priority {
				t.Errorf("priority = %d, want %d", d.Priority, tc.priority)
			}
			if d.TTL != tc.ttl {
				t.Errorf("ttl = %v, want %v", d.TTL, tc.ttl)
			}
			if d.Scope != tc.scope {
				t.Errorf("scope = %q, want %q", d.Scope, tc.scope)
			}
		})
	}
}

func TestEffectiveTTL(t *testing.T) {
	// No lease -> DefaultTTL (a directive is never persistent).
	if got := EffectiveTTL(0); got != DefaultTTL {
		t.Errorf("EffectiveTTL(0) = %v, want DefaultTTL %v", got, DefaultTTL)
	}
	if got := EffectiveTTL(-5 * time.Second); got != DefaultTTL {
		t.Errorf("EffectiveTTL(negative) = %v, want DefaultTTL %v", got, DefaultTTL)
	}
	// Over the cap -> MaxTTL (no unbounded pin).
	if got := EffectiveTTL(time.Hour); got != MaxTTL {
		t.Errorf("EffectiveTTL(1h) = %v, want MaxTTL %v", got, MaxTTL)
	}
	// In range -> unchanged.
	if got := EffectiveTTL(3 * time.Second); got != 3*time.Second {
		t.Errorf("EffectiveTTL(3s) = %v, want 3s", got)
	}
}
