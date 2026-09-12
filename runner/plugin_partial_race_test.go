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
	"os"
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
)

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
// numTurns is how many turns each test drives.
const numTurns = 2

// reproducerEnv gates the truncated-stream reproducer out of the default run.
//
// It detects the race in roughly 1 run in 10 under the suite's own command, and
// the defect it finds is pre-existing, so leaving it on would fail unrelated
// changes for something they did not cause while a green run would still prove
// nothing. Run it deliberately instead. The control case below stays on: it is
// deterministic and it is what keeps the reproducer's scope honest.
const reproducerEnv = "ADK_RUN_RACE_REPRODUCER"

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
// This is a reproducer, not a gate: a green run means nothing. The producing
// goroutine usually returns before it reaches the read — the consumer has
// already stopped, so `if !yield(ev, nil) { return }` wins — and only the
// interleaving that gets there records the pair. Measured on darwin/arm64,
// n=10 each, with the reproducer enabled:
//
//	-run 'TestPluginPath' -race -count=1        detected 8/10
//	-race -count=1 -shuffle=on (whole package)  detected 2/10
//
// The second is the suite's own command, so a passing suite says nothing about
// whether this is fixed — which is why the reproducer is skipped by default.
// Instrumenting base_flow to log whether the read on a partial executes
// confirms the cause rather than leaving it to inference: across 7 green runs
// it never ran, and in the 1 red run it ran twice. Read the mechanism above
// rather than trusting a run.
func TestPluginPathIsRaceFreeOnATruncatedStream(t *testing.T) {
	if os.Getenv(reproducerEnv) != "1" {
		t.Skipf("diagnostic reproducer, not a gate: set %s=1 to run it\n"+
			"\tADK_RUN_RACE_REPRODUCER=1 go test -race -mod=readonly ./runner -run TestPluginPath -count=1",
			reproducerEnv)
	}

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
