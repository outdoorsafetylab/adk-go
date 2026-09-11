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
	"google.golang.org/adk/v2/plugin/loggingplugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// streamingModel emits a partial ("streaming") event, which every real streaming
// backend does — model/gemini's generateStream runs responses through
// NewStreamingResponseAggregator, which marks each one Partial. The scriptedModel
// the existing plugin tests drive never emits one.
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

func runWithPlugins(t *testing.T, appName string, plugins ...*plugin.Plugin) {
	t.Helper()
	const userID, sessionID = "u", "s"

	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &streamingModel{}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := New(Config{
		AppName:           appName,
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for _, q := range []string{"q1", "q2"} {
		for range r.Run(t.Context(), userID, sessionID, genai.NewContentFromText(q, genai.RoleUser), agent.RunConfig{}) {
		}
	}
}

// TestPluginPathIsRaceFreeOnPartialEvents pins that handing an event to the
// plugin callback does not race the node goroutine that produced it.
//
// The scheduler gives every non-partial event a back-pressure handshake: the
// producer blocks on eventItem.processed until the consumer has yielded and
// persisted it (workflow/scheduler.go). Partial events are deliberately
// fire-and-forget and get no handshake, so the producing goroutine runs on while
// the consumer is still inside yield.
//
// fromPlugin writes Actions.Compaction on that same event inside yield, and
// IsFinalResponse reads Actions.Compaction from the producing goroutine, so on a
// partial event the two are unsynchronized. Both sides arrived in v2.3.0; v2.2.0
// has neither, and this test passes there.
//
// The plugin here is the shipped loggingplugin, whose OnEventCallback is the
// ordinary observing hook: it reads the event and returns nil.
func TestPluginPathIsRaceFreeOnPartialEvents(t *testing.T) {
	lp, err := loggingplugin.New("logging_plugin")
	if err != nil {
		t.Fatalf("loggingplugin.New() error = %v", err)
	}
	runWithPlugins(t, "builtin_plugin_partial", lp)
}

// TestPartialEventReachesPluginCallback guards the test above: if partial events
// ever stop reaching plugin callbacks, that test would pass for a reason
// unrelated to the race it is meant to pin.
func TestPartialEventReachesPluginCallback(t *testing.T) {
	var partials atomic.Int64
	observer, err := plugin.New(plugin.Config{
		Name: "observer",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.LLMResponse.Partial {
				partials.Add(1)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	runWithPlugins(t, "observer_partial", observer)
	if partials.Load() == 0 {
		t.Fatal("no partial event reached OnEventCallback; the race test above no longer exercises the unhandshaked path")
	}
}
