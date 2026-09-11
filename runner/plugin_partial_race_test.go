// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runner

import (
	"context"
	"iter"
	"strings"
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

// truncatedStreamModel emits a partial ("streaming") event and then stops
// without the terminal non-partial aggregate.
//
// That is the shape model/gemini's generateStream produces when the upstream
// stream ends early: both the mid-stream error path and the consumer-stopped
// path return before aggregator.Close(), so the chunks already yielded are the
// whole turn. A stream that runs to completion instead ends on the aggregate,
// and that event is handshaked — see TestPluginPathOnCompletedStream.
type truncatedStreamModel struct{ emitted atomic.Int64 }

func (m *truncatedStreamModel) Name() string { return "truncated-stream" }

func (m *truncatedStreamModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.emitted.Add(1)
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("chunk", "model"),
			Partial: true,
		}, nil)
	}
}

// completedStreamModel emits a partial chunk and then the terminal aggregate,
// which is what a stream that runs to completion looks like.
type completedStreamModel struct{ emitted atomic.Int64 }

func (m *completedStreamModel) Name() string { return "completed-stream" }

func (m *completedStreamModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.emitted.Add(1)
		if !yield(&model.LLMResponse{
			Content: genai.NewContentFromText("chunk", "model"),
			Partial: true,
		}, nil) {
			return
		}
		m.emitted.Add(1)
		yield(&model.LLMResponse{Content: genai.NewContentFromText("chunk answer", "model")}, nil)
	}
}

// partialCounter rides alongside the plugin under test in the SAME run, so what
// it counts is evidence about that run rather than about a separate one.
func partialCounter(t *testing.T, partials *atomic.Int64) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "partial_counter",
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
	return p
}

// runTurns drives two turns and fails on any error the runner reports. An
// unchecked error would let this file pass for the wrong reason: when
// OnEventCallback returns an error, run_node.go skips fromPlugin entirely, so
// the write under test never happens.
func runTurns(t *testing.T, appName string, m model.LLM, plugins ...*plugin.Plugin) {
	t.Helper()
	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m})
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
		for _, runErr := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(q, genai.RoleUser), agent.RunConfig{}) {
			// One error is expected and orthogonal: a turn whose last event is a
			// partial trips base_flow's "last event is not final" check on v2.2.0
			// and v2.3.0 (the check is gone by v2.4.0). Everything else fails the
			// test, because an unchecked error would let this file pass for the
			// wrong reason — when OnEventCallback returns an error, run_node.go
			// skips fromPlugin and the write under test never happens.
			if runErr != nil && !strings.Contains(runErr.Error(), "last event is not final") {
				t.Fatalf("Run() error = %v", runErr)
			}
		}
	}
}

// TestPluginPathIsRaceFreeOnATruncatedStream pins that handing an event to the
// plugin callback does not race the node goroutine that produced it.
//
// The scheduler gives every non-partial event a back-pressure handshake: the
// producing goroutine blocks on eventItem.processed until the consumer has
// yielded and persisted it (workflow/scheduler.go). Partial events are
// deliberately fire-and-forget and get no handshake, so the producer runs on
// while the consumer is still inside yield.
//
// fromPlugin writes Actions.Compaction on that event inside yield
// (runner/runner.go), and the producing goroutine reads Actions.Compaction
// through IsFinalResponse on the turn's last event
// (internal/llminternal/base_flow.go). When the last event is a partial, the two
// are unsynchronized.
//
// Both sides arrived in v2.3.0; v2.2.0 has neither and this file passes there.
//
// The plugin under test is the shipped loggingplugin, whose OnEventCallback
// reads the event and returns nil — the ordinary observing hook, and the branch
// of fromPlugin that writes back onto the caller's event.
func TestPluginPathIsRaceFreeOnATruncatedStream(t *testing.T) {
	lp, err := loggingplugin.New("logging_plugin")
	if err != nil {
		t.Fatalf("loggingplugin.New() error = %v", err)
	}
	var partials atomic.Int64
	m := &truncatedStreamModel{}
	runTurns(t, "truncated_stream", m, lp, partialCounter(t, &partials))

	if got := m.emitted.Load(); got == 0 {
		t.Fatalf("model emitted %d responses; the run did not happen", got)
	}
	if partials.Load() == 0 {
		t.Fatal("no partial event reached OnEventCallback in this run; this test no longer exercises the unhandshaked path")
	}
}

// TestPluginPathOnCompletedStream is the other half of the boundary: the same
// plugins and the same partial chunk, but the stream ends on the terminal
// aggregate. That event is handshaked, so it is the last event the producer
// reads and there is no race. It is what pins the scope of the test above —
// without it, "partial events race" would read as a claim about all streaming.
func TestPluginPathOnCompletedStream(t *testing.T) {
	lp, err := loggingplugin.New("logging_plugin")
	if err != nil {
		t.Fatalf("loggingplugin.New() error = %v", err)
	}
	var partials atomic.Int64
	m := &completedStreamModel{}
	runTurns(t, "completed_stream", m, lp, partialCounter(t, &partials))

	if partials.Load() == 0 {
		t.Fatal("no partial event reached OnEventCallback; this test is not comparable to the truncated-stream case")
	}
}
