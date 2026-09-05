# RTX 4090 NVFP4 throughput review

Reviewed on 2026-09-05: the fork's serving checkout at `3a67403`, upstream
`af71ba4`, and the optimization branches linked below. Scope: expert residency,
CPU/GPU dispatch, prefill transfers, HOT adaptation, and diagnostic overhead.
The local checkpoint is Qwen Flash, with `qwen4_exp_text`, 48 layers, 512
experts per layer, top-10 routing, H=2560 and I=640. `qwen3.6-27b` is its API
alias. Measurements use node-4's RTX 4090 and Ryzen 7 7800X3D with 61.91 GiB
RAM and local NVMe. GLM is outside this performance qualification.

## Combined wall time against the original checkout

The direct comparison of original `3a67403` against combined `4429203`
measured eight requests per mode across four isolated starts in
baseline/optimized/optimized/baseline order. Both used CPU DISK prefill,
automatic HOT adaptation, identical cache geometry, and disabled diagnostics,
GPU timing, HOT persistence, and KV reuse. This comparison includes both
telemetry gates and CPU input reuse, but does not activate staged GPU prefill.

| Measured whole response | Original | Combined | Wall-time reduction |
| --- | ---: | ---: | ---: |
| JSON, mean of four requests | 42.091 s | 41.925 s | 0.4%, effectively flat |
| Prose, mean of four requests | 35.003 s | 27.994 s | 20.0% |
| Fixed mix, total of eight requests | 308.374 s | 279.673 s | 9.3% |

This aggregate is provisional. The first matched start pair was 3.2% slower;
the reversed pair was 19.3% faster. Three of four pairs improved for each
workload, but run order and retained host page cache remain material limits.
The result does not establish a dependable general combined gain and cannot
be added to the separate per-change percentages below.

All twelve JSON responses, including warmups, completed normally and passed
integer-value, key-order, and multiplicity checks. Two measured optimized
responses used extra formatting, producing 449 tokens instead of 383. The
whole-task times include those tokens; token throughput is not a substitute
for task completion time. Prose used 192 output tokens in both modes and
remains unscored. All four starts scored 7/8 on the long fidelity questions;
the same code-trace question failed, with original answer `108` and combined
answer `100`, both incorrect against `68`.

The [complete record](../bench/results/4090-combined-cpu-wall-20260905.json)
retains the exact driver, clients, source and native binary identities,
worker mappings, all outputs, journals, and reproducible analysis. Startup
checks confirm equal PIN/DISK placement, HOT capacity, activation dtype,
adaptation settings, cache slots, KV pages, and fetch reserve. The original
service was restored and verified with a real completion.

## Current bottlenecks

The main remaining problem is the interaction between prefill and decode
residency. The measured placement pins 26.44 GiB of expert weights, leaves
37.02 GiB in file-backed DISK banks, and protects 82 experts per DISK layer
in VRAM. Long prefill touches a large expert union, while the following
decode needs a smaller, changing working set. Reducing first-token time can
therefore make the rest of the response slower.

The separate CPU-versus-staged comparison makes this visible. Selected
GPU staging with HOT reuse reduced JSON first-token time from 11.521 to
7.573 seconds, but subsequent decode grew from 15.086 to 17.322 seconds.
Whole-response JSON time improved by 6.4% on average with only two of four
pairs faster. Prose improved by 31.6%, with all four pairs faster. Worker
storage-read accounting also increased for JSON, from 3.566 to 4.484 GiB per
response. Those counters include other worker I/O, so they support further
residency investigation without attributing a specific eviction cause. See
the [complete direct comparison](../bench/results/4090-staged-hot-reuse-cpu-auto-20260905.json).

A separate diagnostic replay found approximately 11% fixed-HOT decode
coverage versus approximately 97% retrospective coverage with the same
capacity in each logged window. That oracle can see future requests within
its window; it is an opportunity bound, not an achievable online hit rate.
PLE gather was approximately 0.5 ms per step in most logged requests. The
evidence directs the next investigation toward expert residency and cold
execution. Diagnostic wall times are excluded from performance claims.

The automatic adaptation cadence is another concrete mismatch to inspect.
After filling, it advances to a 1,000-routed-token interval shared by prefill
and decode. In each CPU-versus-staged start, journals show 27 prefill dispatches and
three decode dispatches across the full timing and fidelity sequence. A
192- or 383-token reply can finish before the next decode update, depending
on where its prompt ended on that shared clock. This can make adaptation
mostly optimize prefill residency even with split histories and phase aim.
The existing fixed-interval option allowed a controlled cadence experiment
without changing routing, weights, ranking, or swap bounds. A four-start
auto/64/64/auto comparison with CPU prefill increased prose response time
by 34.5%, with every pair slower. Two automatic-mode JSON outputs reached
the benchmark's 384-token cap before finishing; the two complete JSON pairs
were also slower with fixed-64 cadence. The
[record](../bench/results/4090-hot-cadence-cpu-wall-20260905.json) preserves
those failures. Faster cadence also increases prefill catch-up work and
bypasses automatic backoff, so more frequent updates alone are unqualified.

Other measured waste has concrete fixes: repeated CPU input quantization per
route, serial HOT route preparation, unnecessary short-prefill transfers, and
diagnostic GPU reductions. These improvements preserve the actual routes.

## Review of the fork and upstream

The fork's file-backed HostBanks, protected HOT slots, io_uring PLE, host
memory governor, and CPU DISK execution make the third tier possible. The
hard boundaries are cache ownership, asynchronous publication, scratch
lifetime, and total host memory. The optimization work now covers retired
HOT slots, sparse unowned scratch, cache rebuilding, bounded staging, and
real NVFP4 GEMM output parity.

Upstream's [design paper](https://arxiv.org/html/2608.16157v1#S3) assumes a
CPU-resident expert pool as its source of truth. It overlaps full-layer
prefill transfers with GPU work and splits decode misses between CPU
execution and GPU cache fills using measured host and PCIe bandwidths. Its
shared expert cache and recurrent-state prefix reuse are useful foundations.
Extending that model to DISK requires accounting for fault latency, storage
bandwidth, and page-cache competition. Applying the RAM bandwidth ratio
alone to cold storage misses would omit those costs. This is an inference
from the design and the local measurements, not an upstream benchmark result.

Upstream was refreshed to `af71ba4` during review. The three commits since
the previously inspected `092ce4a` change issue templates, repository
instructions, and wheel publishing, with no engine runtime changes. No
upstream merge is part of these PRs.

The current NOREUSE release path also needs attention before using it as a
cache-control mechanism: it advises a fresh descriptor that performs no
read or mmap. The [kernel source review](disk-prefill-staging.md#cache-advice-review)
explains why the existing mapping does not inherit that flag on upstream
Linux 6.8. Eager release is off in these benchmarks, so this finding does
not explain the CPU/staged difference.

## Changes and evidence

PR states below describe this review date. Percentages are time reductions;
kernel speedups and model wall time have separate meanings.

| Change | Evidence | Delivery |
| --- | --- | --- |
| Selective short-prefill transfers and diagnostic gating | About 15% lower 76-token TTFT in the initial acceptance run; larger prompts were mixed | [#27](https://github.com/jomcgi-org/freetoken-fork/pull/27), ready |
| Parallel HOT route preparation | 20,480-route kernel: 1,719.7 to 39.3 microseconds; model-wide gain remains unquantified | [#28](https://github.com/jomcgi-org/freetoken-fork/pull/28), ready |
| Concurrent CPU/GPU prefill | Ordering and placement confounded the early model gain | [#29](https://github.com/jomcgi-org/freetoken-fork/pull/29), draft |
| Reuse CPU input quantization across routes | Whole-request reductions of 5.0%, 11.4%, 16.6% at 76, 524, 2,060 prompt tokens; all 48 prefill pairs faster and all 56 timing pairs identical in text and usage | [#30](https://github.com/jomcgi-org/freetoken-fork/pull/30), ready |
| Gate diagnostic-only decode classification and idle history readback | Removes ordinary decode's reduction/counter work and idle coverage's GPU-to-CPU history synchronization; 99 focused Linux tests pass, with no independent wall-time claim | [#31](https://github.com/jomcgi-org/freetoken-fork/pull/31), ready; also included in #33 |
| Correct temporary materialization ownership | Prevents sparse scratch from advertising uncopied experts as hits; protected HOT bytes retained | [#32](https://github.com/jomcgi-org/freetoken-fork/pull/32), ready |
| Selected DISK staging and exact HOT VRAM reuse | Strong long-prefill gains, 31.6% lower prose response time in the latest CPU comparison, mixed JSON response-time results | [#33](https://github.com/jomcgi-org/freetoken-fork/pull/33), draft |
| Experimental direct-only staged reads | Exact byte/GEMM checks pass, but the fixed response mix is 1.3% slower, with six of eight pairs slower and substantially more storage reads | [#34](https://github.com/jomcgi-org/freetoken-fork/pull/34), draft; serving remains buffered |

The main dependency chain is #27, #28, #30, #32, #33. #31 is independently
reviewable from #28 and is cherry-picked into #33. #29 remains separate.
#34 evaluates a transport option on top of #33.
The original serving checkout is restored after every model gate; these
changes are delivered in PRs.

## Quality and measurement contract

Preserve checkpoint bytes, scale bytes, selected expert IDs, router weights,
and every routed contribution. Reuse data and change where work executes
without giving cache residency any influence over the router's selections.
CPU input reuse copies the existing quantized activation bytes and scales;
it introduces no new quantization. HOT staging reuse copies existing GPU
weight bytes and leaves the GPU GEMM unchanged.

Node-4 uses BF16 GPU activations because native NVFP4 activation mode requires
SM120; the 4090 is SM89. CPU and GPU execution already have different numeric
paths. Switching placement can therefore change generated text. The transport
tests compare exact bytes and BF16 GEMM output bits, while model checks compare
scored tasks and complete responses. In the CPU-versus-staged gate, all JSON
records were correct and both modes retained the same 7/8 long-fidelity score,
including the same wrong code-trace answer. The combined comparison also
retained 7/8 but changed that incorrect answer. Prose remains unscored;
this sample does not establish broad model-quality equivalence.

Use `--moe-collect-stats` and `--moe-step-timing` only for diagnostic runs.
Functional HOT histories, session-prefetch observations, and the WILLNEED
fault guard still run when diagnostics are off. The selected-union host
readback is required transport work. Optional `/proc` I/O snapshots run
outside the client timer. Qualify performance with complete responses,
matching placement and token counts, full warmups, and reversed request
orders. The staged wall measurements precede the final #31 cherry-pick;
the combined comparison includes it. No independent throughput percentage
is claimed for that gate.

## Next performance gate

Keep CPU DISK prefill as the default and use staged execution as an opt-in
for the measured long-prefill workloads. The next larger opportunity is
protecting useful decode residency while reducing prefill's storage traffic.
The [direct-only staging experiment](direct-prefill-io.md#whole-response-result-direct-only-reads-do-not-qualify)
is now measured: JSON takes 4.8% longer, while prose averages 3.3% shorter
with only one of four pairs faster in each workload. Worker storage reads
increase from about 5-6 GiB to about 27 GiB per response. The fixed mix is
1.3% slower overall, so the reader remains an experimental comparison point.

The next candidate should retain useful RAM cache hits while reducing
insertion of cold prefill data. Placement budgets and reuse of resident
weights remain possible improvements, with the router and checkpoint fixed.
Each needs a fresh whole-response comparison with JSON and prose, plus the
same quality checks. A nonblocking buffered read is insufficient proof of
avoiding cache insertion on this kernel; the direct-reader document records
the source review and constraints on residency hints.

The tested automatic decode-history preference was reverted: it improved
JSON by 26.8% but slowed prose by 8.6%. That experiment changed only placement
planning, yet still failed the general throughput gate. It provides no reason
to bias the model toward HOT experts.
