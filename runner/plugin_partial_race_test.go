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
	"fmt"
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/loggingplugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// numTurns is how many turns each test drives.
const numTurns = 2

// truncatedStreamModel completes a turn with a partial ("streaming") event as
// its last one: it yields one partial, then its iterator returns normally with
// no error and no terminal aggregate.
//
// That is the condition the racing read needs. base_flow.Run reads
// lastEvent.IsFinalResponse() only after the inner iterator finishes normally —
// an error and a stopped consumer both return before it — so the last event
// reaching that read is a partial exactly when a model ends a turn this way.
//
// model/gemini does not: its generateStream appends aggregator.Close() on clean
// completion, and Close returns a response whenever any chunk was processed, so
// a partial there is always followed by the non-partial aggregate. That shape is
// TestPluginPathOnCompletedStream, and it does not race. What this test
// describes is therefore the model.LLM contract, which permits ending a turn on
// a partial, rather than any behaviour of the shipped Gemini backend.
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
// which is what model/gemini produces on clean completion.
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

// eventTally records the shape of what actually reached a plugin callback.
type eventTally struct{ partial, nonPartial atomic.Int64 }

// eventCounter rides alongside the plugin under test in the SAME run, so what it
// counts is evidence about that run rather than about a separate one.
func eventCounter(t *testing.T, tally *eventTally) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "event_counter",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev.LLMResponse.Partial {
				tally.partial.Add(1)
			} else {
				tally.nonPartial.Add(1)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// runTurns drives numTurns turns and fails on any error the runner reports. An
// unchecked error would let this file pass for the wrong reason: when
// OnEventCallback returns an error, run_node.go skips fromPlugin entirely, so
// the write under test never happens.
//
// allowNotFinal belongs to the truncated-stream case alone. A turn whose last
// event is a partial trips base_flow's "last event is not final" check on v2.2.0
// and v2.3.0 (the check is gone by v2.4.0). A completed stream must never need
// that exemption, so granting it there would hide an aggregate that went
// missing — which is the whole thing the control case exists to rule out.
func runTurns(t *testing.T, appName string, m model.LLM, allowNotFinal bool, plugins ...*plugin.Plugin) {
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
	for i := range numTurns {
		q := fmt.Sprintf("q%d", i+1)
		for _, runErr := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(q, genai.RoleUser), agent.RunConfig{}) {
			if runErr == nil {
				continue
			}
			if allowNotFinal && strings.Contains(runErr.Error(), "last event is not final") {
				continue
			}
			t.Fatalf("Run() error = %v", runErr)
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
//
// Without the base_flow change this commit makes, detection was probabilistic
// and therefore not a gate: the producing goroutine usually returned before it
// reached the read (the consumer had already stopped, so
// `if !yield(ev, nil) { return }` won), and only the interleaving that got there
// recorded the pair. Measured on darwin/arm64, n=10 each:
//
//	                                            before  after
//	-run 'TestPluginPath' -race -count=1         8/10    0/10
//	-race -count=1 -shuffle=on (whole package)   2/10    0/10
//
// With the read hoisted above the handoff there is no unsynchronized access
// left, so the green is deterministic and this belongs in the default run.
//
// Be clear about what it is worth in the other direction, though: the "before"
// column IS the measurement of moving the read back below the yield, so as a
// regression detector this is probabilistic, not a gate — 2 runs in 10 under the
// suite's own command. It documents the invariant and it cannot go red for any
// other reason; it will not reliably catch someone undoing it.
func TestPluginPathIsRaceFreeOnATruncatedStream(t *testing.T) {
	lp, err := loggingplugin.New("logging_plugin")
	if err != nil {
		t.Fatalf("loggingplugin.New() error = %v", err)
	}
	var tally eventTally
	m := &truncatedStreamModel{}
	runTurns(t, "truncated_stream", m, true, lp, eventCounter(t, &tally))

	if got := m.emitted.Load(); got == 0 {
		t.Fatalf("model emitted %d responses; the run did not happen", got)
	}
	if tally.partial.Load() == 0 {
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
	var tally eventTally
	m := &completedStreamModel{}
	runTurns(t, "completed_stream", m, false, lp, eventCounter(t, &tally))

	// Both halves are load-bearing. Without the partial this is not comparable to
	// the truncated-stream case; without the terminal aggregate there is nothing
	// handshaked to explain why it does not race, and an aggregate that went
	// missing would leave this passing for the wrong reason.
	if tally.partial.Load() == 0 {
		t.Fatal("no partial event reached OnEventCallback; this test is not comparable to the truncated-stream case")
	}
	if got := tally.nonPartial.Load(); got < numTurns {
		t.Fatalf("non-partial events reaching OnEventCallback = %d across %d turns, want at least %d — the terminal aggregate is what makes this case handshaked, so without it this test proves nothing",
			got, numTurns, numTurns)
	}
}

// skipSummarizationModel yields a function call on the first turn and text
// afterwards, counting how many times it was asked.
type skipSummarizationModel struct {
	calls atomic.Int64
}

func (m *skipSummarizationModel) Name() string { return "skip-summarization" }

func (m *skipSummarizationModel) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	first := m.calls.Add(1) == 1
	return func(yield func(*model.LLMResponse, error) bool) {
		if first {
			yield(&model.LLMResponse{Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "noop"}}},
			}}, nil)
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", "model")}, nil)
	}
}

// TestNonPartialStopConditionIsReadAfterTheCallbacks pins where the stop
// condition for a NON-partial event is decided: after the consumer's plugin
// callbacks, not before.
//
// The partial case has to be decided before the handoff, because a partial is
// not handshaked and reading it afterwards races fromPlugin's write. It is
// tempting to hoist the non-partial read the same way. This test is why that is
// wrong: IsFinalResponse also reads Actions.SkipSummarization, fromPlugin
// restores Compaction only, and so a plugin that sets SkipSummarization in place
// keeps it. A non-partial event IS handshaked, so the late read sees that and
// stops the loop; an early read would see the pre-callback false, run the tool
// response through another model call, and silently undo SkipSummarization.
//
// The differential is the model call count: 1 when the read is late, 2 when it
// is early.
func TestNonPartialStopConditionIsReadAfterTheCallbacks(t *testing.T) {
	noop, err := functiontool.New(functiontool.Config{
		Name:        "noop",
		Description: "returns a value",
	}, func(_ agent.Context, _ struct{}) (string, error) {
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	var set atomic.Int64
	skipper, err := plugin.New(plugin.Config{
		Name: "skipper",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			// In place, returning nil — the idiom fromPlugin documents, and
			// SkipSummarization is a field it does not restore.
			if ev.LLMResponse.Content != nil {
				for _, part := range ev.LLMResponse.Content.Parts {
					if part != nil && part.FunctionResponse != nil {
						ev.Actions.SkipSummarization = true
						set.Add(1)
					}
				}
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}

	m := &skipSummarizationModel{}
	root, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m, Tools: []tool.Tool{noop}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := New(Config{
		AppName:           "skip_summarization",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      PluginConfig{Plugins: []*plugin.Plugin{skipper}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for _, runErr := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("call the tool", genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("Run() error = %v", runErr)
		}
	}

	if set.Load() == 0 {
		t.Fatal("the plugin never saw a function response event; this test is not exercising SkipSummarization")
	}
	if got := m.calls.Load(); got != 1 {
		t.Errorf("model calls = %d, want 1 — SkipSummarization set by a plugin on the tool response must stop the loop, which only holds if the non-partial stop condition is read after the callbacks", got)
	}
}
