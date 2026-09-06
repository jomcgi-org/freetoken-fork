# Concurrent complete-response qualification

The completed four-start comparison does not establish a dependable throughput
gain from four offered requests. Across twenty-four measured responses per mode,
total workload elapsed time is 498.127 seconds at concurrency one and 512.904
seconds at concurrency four, a 3.0% increase. The first matched order is 18.7%
slower with four requests; the reverse order is 12.1% faster. Keep this aggregate
qualified by the disagreement between orders and retained serving state.

| Measured quantity | One offered request | Four offered requests |
| --- | ---: | ---: |
| Fixed 24-response mix, elapsed | 498.127 s | 512.904 s |
| Mean individual response latency | 20.755 s | 82.122 s |
| Mean complete JSON latency | 24.567 s | 97.694 s |
| Mean complete prose latency | 16.942 s | 66.549 s |
| Whole-worker storage reads during measurement | 53.638 GiB | 93.916 GiB |

These are group elapsed times, not sums of overlapping request durations.
All starts have identical placement, 3,920 expert-cache slots, 1,024 KV pages,
a 1,024-row fetch reserve, and CUDA graph sizes 1/2/4. Actual status logs show
four-request decode batches in both concurrent starts. Server capacity four is
fixed in both modes; this is not a comparison against the previous production
server's single-request capacity or its 4,045-slot cache geometry.

All thirty-two JSON outputs, including warmups, complete normally and pass exact
value, type, key-order, and multiplicity checks. All twelve measured JSON pairs
have identical text and usage. All thirty-two prose responses complete with
three paragraphs; two measured prose pairs have identical text and usage. Every
start scores 7/8 on fidelity with identical answers, including the existing
`108` code-trace failure against expected `68`.

Assistant review against the supplied reference finds explicit coverage of
82/84 measured prose constraints at concurrency one and 83/84 at concurrency
four. Warmup coverage is 27/28 and 25/28 respectively. Both modes contain
omissions and unsupported connections, especially between file hints and HOT
ownership. No direct core contradiction was identified in this small audit;
the format checks and coverage counts do not establish broad quality equivalence.

The concurrent starts differ substantially: 289.027 versus 223.877 seconds,
with 63.630 versus 30.286 GiB of measured worker storage reads. These counters
include prefill, decode, PLE, and adaptation, so they cannot isolate the cause.
Actual adaptation transitions also differ despite identical settings. The final
single-request start backs off its interval floor from 1,000 to 2,000 after a
746 ms staging batch; the other starts do not. These observations constrain a
causal or general performance claim.

The [complete record](../bench/results/4090-concurrent-wall-20260906.json) retains
all outputs, prompt manifests, group summaries, journals, source/native identities,
review notes, exact drivers, and reproducible analysis. It verifies timing and
concurrency bounds, output checks, equal geometry, terminal driver success, and
a real completion from the restored original system service. The experiment is
complete; concurrent serving has not qualified as a throughput improvement.

The sustained reader gates measure one request at a time. The
[`concurrent-prefill-wall.py`](../bench/concurrent-prefill-wall.py) client
uses the same deterministic JSON and reference-based prose prompts to
measure a fixed workload with bounded client concurrency. It continuously
refills completed requests in manifest submission order. All warmup responses
finish before any measured request is submitted.

The primary metric is the elapsed time for the entire measured workload,
from first submission through final response recording and worker cleanup.
Summed request wall time is retained separately and is not aggregate elapsed
time. Each response retains its full text, usage, finish reason, streaming
latencies, client queue time, and offsets within the group. A transport error
is retained as a failed record; remaining requests still run. Checked rates
are omitted when any response fails completion or output checks. Prose
formatting is separate from semantic review and broad quality equivalence.

Whole-worker I/O is sampled only before and after each group, outside the
timer. It cannot be attributed to concurrent individual requests. There are
no per-request I/O snapshots or first-text diagnostics. The CLI refuses to
overwrite an earlier output, prompt manifest, or summary.

A model experiment must keep the runtime, reader policy, precision, prompts,
cache geometry, and server admission limit fixed while varying offered
concurrency. Record actual server arguments, native identities, startup
geometry, adaptation transitions, and retained state. Use both start orders,
disable diagnostic stats and GPU timing, and restore the original system
service after the comparison. Report aggregate wall time and individual
latency together. A throughput gain with slower individual responses is a
tradeoff, not a single-request speedup. This client introduces no serving
policy or model changes, and the completed gate has not qualified concurrent serving on
the 4090. The HOT staging reader gate is separate and unchanged.

All eighteen targeted pure Python checks pass locally and on Linux without
a model or network service. They cover actual overlapping workers, the concurrency
bound, refilling while an earlier response remains pending, error retention,
warmup separation, exact JSON checks, prose format versus semantics, and
workload timing that includes recording but excludes outer I/O snapshots.
The [Linux validation record](../bench/results/4090-concurrent-client-validation-20260906.json)
retains the exact source hashes and command at `c0775ea`. It verifies that the
HOT reader wall-time driver finished successfully before testing and that
the original service remained active with the same PID. These eighteen checks
qualify the client protocol; the model comparison above is separate evidence.

The checked-in [model driver](../bench/concurrent-wall-driver.py) freezes
`c0775ea` and runs four starts with offered concurrency 1/4/4/1. Every start
uses the same four-request admission limit, CUDA graph capacity four, mmap
HOT staging, buffered GPU DISK prefill, and automatic HOT adaptation settings.
The existing KV ladder only supports admission capacity one, so this gate
explicitly disables it and reserves 65,536 KV tokens in both modes. This
matches the earlier pool floor, but the four-request state allocation may
change other startup budgets. Verify identical geometry within this gate;
do not combine its percentages with the earlier single-capacity results.

Each start has four complete warmups, twelve measured complete responses,
and eight fidelity cases run at that start's offered concurrency. Fidelity
backgrounds come from manifest order, independent of response completion
order. Whole-worker snapshots bracket each client group. Failed responses
remain in the record; a client output-check failure does not skip the
opposite mode when all scheduled records and group summaries are present.
Report every failure and semantic-review limitation before qualifying a gain.

Concurrent response windows overlap. Source-balanced early/late and per-kind
latencies may describe behavior, but their sums are not separate elapsed
workload times. Pair JSON and prose by prompt identity, compare both start
orders, and retain per-request latency alongside the full measured-group
duration. This does not qualify production prefix reuse, idle convergence,
very long contexts, or a single-request speedup from batching.

The driver checks actual CPU executor and graph capacity at startup, native
worker mappings, transport settings, and clean frozen sources. `--preflight`
is read-only. Launch its detached unit with a two-hour runtime limit and
the [recovery script](../bench/concurrent-wall-restore.sh) as `ExecStopPost`.
Each server start has a 45-minute limit. The normal finalizer restores the
original service and verifies a real completion. Observe the live driver
after a session interruption; never restart it merely because observation
timed out.

The driver and recovery script pass syntax checks. The
[Linux preflight](../bench/results/4090-concurrent-wall-preflight-20260906.json)
confirms identical frozen revisions and native identities, the fixed
four-request settings, and a driver hash matching the committed script.
It does not measure concurrent model performance.
