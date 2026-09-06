# Multi-turn Pi task benchmark

Independent prompts omit much of interactive agent latency: repeated tool calls,
growing history, prefix reuse, local tests, and repairs after those tests fail.
This benchmark runs the real Pi CLI against the local FreeToken endpoint and
measures time from the first user prompt through independent verification of
the final task. Pi startup is reported separately. Test execution and repair
prompts remain inside the task clock. Artifact writes and final session export
are outside it.

The first fixture is a small pure Python cache. In one continuing conversation,
the agent fixes TTL boundaries and invalidation, adds capacity and LRU behavior,
then adds a cached loader with exception semantics. Every stage must pass
independent cumulative checks. A failing check is returned to the same session,
with a fixed repair budget. Grading lives outside the writable fixture and a
snapshot is taken before a run. The agent may edit its own tests; changing those
tests cannot weaken the independent checks. This is a small smoke workload,
not a general coding-quality evaluation.

The client uses [Pi RPC](https://pi.dev/docs/latest/rpc), waiting for
`agent_settled` after each prompt. `agent_end` alone can precede automatic
continuations. Read, bash, edit, and write are enabled. User configuration,
extensions, skills, context files, telemetry, and startup network discovery are
disabled. Automatic transport retry and context compaction are disabled so that
an error or overflowing history is retained as a failed attempt. Test-repair
turns are enabled explicitly by the harness. This configuration isolates client
state; it is not an operating-system sandbox for shell commands. Run it in a
disposable environment when expanding beyond these trusted synthetic fixtures.

Install the pinned Pi version and dependency lock, separately from model setup:

```sh
npm ci --prefix bench/pi --no-audit --no-fund
python3 bench/pi-agentic-wall.py \
  --pi "$PWD/bench/pi/node_modules/.bin/pi" \
  --base-url http://127.0.0.1:8090/v1 \
  --label baseline-integration \
  --output-dir /tmp/pi-agentic-baseline \
  --sessions 1
```

Node 22.19 or newer is required by the pinned Pi package. The harness itself
uses the Python standard library. Only loopback model endpoints are accepted;
an SSH tunnel can connect the Mac client to the 4090 server. No paid provider
is configured. The API alias `qwen3.6-27b` refers to the deployed Qwen Flash
model in this setup. The [custom model configuration](https://pi.dev/docs/latest/models)
uses OpenAI chat completions, temperature zero, the existing non-thinking
benchmark setting, and a fixed maximum output per model call. Every comparison
must keep those settings identical. Changing thinking mode is a separate
quality experiment.

Each session retains complete messages, final files, stage checks, errors,
tool failures, model usage, model-call count, and wall time. The default path
discards streaming delta events after consuming them. `--trace` retains those
events for diagnosis and marks the record; trace timings cannot establish a
speedup. There is no new server instrumentation. Tool time is the union of
execution spans, avoiding double counting overlapping tools. Model usage comes
from complete assistant messages once, not duplicated agent-end snapshots.

An unsuccessful task keeps its elapsed time and has no verified completion
time. A mixed-success set does not receive a checked tasks-per-hour figure.
Always compare success fraction, all attempted wall time, model calls and
repair count alongside successful-task latency. A quick failure is not a win.
Timeouts, truncated responses, RPC errors, and exhausted repair budgets remain
visible. Pi can recover a truncated intermediate tool call within the same
budget; a truncated final response cannot qualify a stage. Per-task defaults
are 900 seconds, 30 model calls, 8192 output tokens per model call, and one
check-driven repair per stage. These bounds must be the same in both arms.

For runtime qualification, use isolated A/B/B/A server starts, repeated complete
sessions, identical server capacity and KV reservation, and diagnostics disabled.
Capture revision, native binary identity, actual flags, model identity, cache
policy and isolation evidence with `--server-metadata`. The client stores that
record but cannot prove exclusivity or that diagnostics are off. It deliberately
does not qualify a performance comparison by itself. A server-controlled driver
and review of both arms must do that.

Run the CLI and fixture tools on the same client host in all arms, with the same
Pi installation path. Pi includes its working directory in the system prompt;
for exact replay of fresh-session prefixes, use an identical workspace path in
disposable client environments, or archive each output directory and reuse its
path. Do not compare different paths as an exact-prefix experiment.

The first stage starts a fresh Pi session. Later stages retain history and can
reuse server prefixes, but a cache hit is not assumed. `--sessions` resets the
fixture and Pi conversation between tasks while retaining the same workspace
path; it does not flush the model server cache or host page cache. Keep fresh
server starts, repeated sessions, and within-session follow-ups distinct. Pi's
`cacheRead` can be zero when a provider omits cached-token usage, so that field
alone cannot establish a miss. Prefix-cache-on/off and long tool-output/context
sweeps are separate follow-up experiments. The existing independent-prompt
benchmark remains useful for isolating kernel changes.

Validation: the new hermetic tests cover task grading, failure and repair
accounting, inclusive wall timing, real RPC pipe framing, Unicode separators,
settled completion, response truncation, call/time limits, and overlapping tool
spans. Real Pi/FreeToken integration and controlled runtime comparisons are
recorded separately; no agentic speedup is claimed by adding this client.

The [18 focused client checks](../bench/results/4090-pi-agentic-client-validation-20260906.json)
pass on Linux and macOS. The [first development smoke](../bench/results/4090-pi-agentic-development-smoke-20260906.json)
passed the TTL stage, exercised real read/write/edit/bash tools and reported
prefix reuse on subsequent model calls. It failed during stage two when the
1536-token development cap truncated a tool call. Its 313.9 seconds and nine
model calls remain a failed integration attempt, with no speedup claim. The
original server had diagnostics enabled and other requests were not excluded.
The client and grader were still being developed during that attempt. The
final client snapshots its grader and defaults to an 8192-token allowance.
The [frozen-client integration at 460c6ff](../bench/results/4090-pi-agentic-integration-20260906.json)
then passed all three independent stages in 464.073 seconds (7m 44s), using
18 model calls, 16 tool executions, and 5493 output tokens. Stage wall times,
including each verifier, were 125.393, 219.687, and 118.994 seconds. Two failed
local test commands were recovered within Pi's own loop; the independent
verifier needed no additional repair prompts. Reported conversation context
grew to 14406 tokens, with cached prefixes reported on 17 of 18 model calls.
All final stage responses completed normally. The record retains both failed
test outputs, later passing runs, complete messages, and the final source/tests.

The Mac spent about 0.424 seconds executing tools and 0.205 seconds in the
independent verifier across the whole run. The remaining wall time includes
model requests, server queueing, transport, and client orchestration; this
integration does not isolate those components. The original server retained
diagnostics and was not isolated from other clients, so 464 seconds is an
integration observation, not a qualified optimization result. No runtime A/B
comparison has been made with this client yet.

The final client also catches interruption, terminates Pi and its remaining
process group, retains the failed attempt, and does not start another session.
An incomplete or cancelled session schedule exits unsuccessfully. The extra
hermetic cancellation case passes on Linux and macOS; it does not change the
completed integration's prompt, sampling, tool, or grading protocol.
