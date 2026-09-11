package runner

import (
	"context"
	"fmt"
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// streamingModel emits partial ("streaming") events, which every real
// streaming backend does and which the existing scriptedModel never does.
type streamingModel struct{ turns atomic.Int64 }

func (m *streamingModel) Name() string { return "streaming" }

func (m *streamingModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	reply := fmt.Sprintf("answer %d", m.turns.Add(1))
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(reply, "model"),
			Partial: true,
		}, nil)
	}
}

// TestPluginPathIsRaceFreeOnPartialEvents pins that handing an event to the
// plugin callback does not race the node goroutine that produced it.
//
// The scheduler gives every non-partial event a back-pressure handshake: the
// producer blocks on eventItem.processed until the consumer has yielded and
// persisted it (workflow/scheduler.go). Partial events are deliberately
// fire-and-forget and get no handshake, so the producing goroutine runs on
// while the consumer is still inside yield.
//
// fromPlugin writes Actions.Compaction on that same event inside yield, and
// IsFinalResponse reads Actions.Compaction from the producing goroutine, so on
// a partial event the two are unsynchronized.
//
// Both sides arrived in v2.3.0; v2.2.0 has neither.
func TestPluginPathIsRaceFreeOnPartialEvents(t *testing.T) {
	const userID, sessionID = "u", "s"

	var seen atomic.Int64
	observer, err := plugin.New(plugin.Config{
		Name: "observer",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.LLMResponse.Partial {
				seen.Add(1)
			}
			// Returning nil means "I changed nothing", the ordinary way to
			// write an observing hook. It is also the branch of fromPlugin
			// that writes back onto the caller's event.
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &streamingModel{}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	r, err := New(Config{
		AppName:           "plugin_partial_race",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: []*plugin.Plugin{observer}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for _, q := range []string{"q1", "q2"} {
		for range r.Run(t.Context(), userID, sessionID, genai.NewContentFromText(q, genai.RoleUser), agent.RunConfig{}) {
		}
	}

	// Guards the test itself: if partial events stop reaching plugins, this
	// test would pass for a reason that has nothing to do with the race.
	if seen.Load() == 0 {
		t.Fatal("no partial event reached the plugin callback; this test no longer exercises the unhandshaked path")
	}
}
