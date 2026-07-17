package kvcache

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
)

// TestRetentionTrailingFieldsCompat verifies that the simulator's EventData emission
// with retention priority/retain_until correctly decodes through the real v0.9.0 adapter.
func TestRetentionTrailingFieldsCompat(t *testing.T) {
	gomega.RegisterTestingT(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	topic := CreateKVEventsTopic(localhost, "compat-model")
	sub, endpoint := common.CreateSub(ctx, topic)
	defer sub.Close()

	// 1. Setup simulator sender
	eventChan := common.Channel[EventData]{
		Channel: make(chan EventData, 10),
		Name:    "test-event-chan",
		Done:    ctx.Done(),
	}
	pub, err := common.NewPublisher(ctx, endpoint)
	require.NoError(t, err)
	sender := NewKVEventSender(pub, topic, eventChan, 1, 16, time.Millisecond, true, logr.Discard())

	go sender.Run(ctx)

	// 2. Setup real vLLM Adapter (v0.9.0)
	adapter := engineadapter.NewVLLMAdapter()

	// Wait for connection to establish
	time.Sleep(100 * time.Millisecond)

	// 3. Emit a BlockStored event with a retention directive
	p := 100
	ru := float64(time.Now().Add(10*time.Minute).UnixNano()) / 1e9

	eventChan.Channel <- EventData{
		action:      eventActionStore,
		hashes:      []uint64{12345},
		tokens:      []uint32{1, 2, 3},
		priority:    &p,
		retainUntil: &ru,
	}

	// 4. Receive and decode with the real adapter
	msg, err := sub.Recv()
	require.NoError(t, err)
	require.Len(t, msg.Frames, 3)

	rawMsg := &kvevents.RawMessage{
		Topic:   string(msg.Frames[0]),
		Payload: msg.Frames[2],
	}

	_, _, batch, err := adapter.ParseMessage(rawMsg)
	require.NoError(t, err)
	require.Len(t, batch.Events, 1)

	storedEv, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok, "Expected BlockStoredEvent")

	assert.Equal(t, []uint64{12345}, storedEv.BlockHashes)
	// Since v0.9.0 adapter doesn't know about Priority/RetainUntil, it will decode it successfully
	// ignoring the extra fields, proving backward compatibility.
}

// TestUnknownEventTagFailsBatch locks in the constraint behind RFC-0001's "no new event
// types in v1" decision: the deployed v0.9.0 adapter fails the *entire* batch on the first
// unknown event tag, dropping known events alongside it. That is precisely why priority
// changes re-emit BlockStored (an idempotent upsert the adapter already understands) rather
// than introducing a dedicated retention event -- a new tag would silently break every
// co-batched BlockStored on a not-yet-upgraded indexer.
func TestUnknownEventTagFailsBatch(t *testing.T) {
	adapter := engineadapter.NewVLLMAdapter()

	// A well-formed BlockStored (map-encoded, exactly as our sender emits it) ...
	known := &blockStoredEvent{
		kvCacheEvent: kvCacheEvent{Tag: string(kvevents.EventTypeBlockStored)},
		BlockHashes:  []any{uint64(1)},
		TokenIds:     []uint32{1, 2, 3},
		BlockSize:    16,
	}
	knownBytes, err := msgpack.Marshal(known)
	require.NoError(t, err)

	// ... co-batched with an event carrying a tag the v0.9.0 adapter does not recognize.
	unknown := map[string]any{"type": "RetentionDirectiveEvent", "priority": 100}
	unknownBytes, err := msgpack.Marshal(unknown)
	require.NoError(t, err)

	dpRank := 0
	batch := msgpackEventBatch{
		TS:               float64(time.Now().UnixNano()) / 1e9,
		Events:           []msgpack.RawMessage{knownBytes, unknownBytes},
		DataParallelRank: &dpRank,
	}
	payload, err := msgpack.Marshal(batch)
	require.NoError(t, err)

	_, _, parsed, err := adapter.ParseMessage(&kvevents.RawMessage{Topic: "compat", Payload: payload})
	require.Error(t, err, "an unknown event tag must fail the whole batch")
	assert.Empty(t, parsed.Events, "no events should survive a batch containing an unknown tag")
}
