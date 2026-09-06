# RTX 4090 NVFP4 throughput review

Reviewed on 2026-09-05: the fork's serving checkout at `3a67403`, upstream
`af71ba4`, and the optimization branches linked below. Scope: expert residency,
CPU/GPU dispatch, prefill transfers, HOT adaptation, and diagnostic overhead.
The local checkpoint is Qwen Flash, with `qwen4_exp_text`, 48 layers, 512
experts per layer, top-10 routing, H=2560 and I=640. `qwen3.6-27b` is its API
alias. Measurements use node-4's RTX 4090 and Ryzen 7 7800X3D with 61.91 GiB
RAM and local NVMe. GLM is outside this performance qualification.

Performance evidence updated on 2026-09-06 after the completed Pi agentic gate,
concurrency gate, resident-populate component timing and continuation-state review.

## Sustained wall time and the reader regression

The strongest sustained combined result is 17.8% less client wall time for
the fixed workload, about 1.22 times the request throughput. Original
`3a67403` uses CPU DISK prefill; optimized `132b629` uses staged GPU prefill,
published HOT reuse, and the cache-aware reader. Inference code is unchanged
from `78848ce`. Four original/optimized/optimized/original starts each run
four warmups, twelve measured complete responses, and eight fidelity cases.
Diagnostics, GPU timing, HOT persistence, and KV reuse are off.

| Measured client response | Original CPU | Combined cached staging | Less wall time |
| --- | ---: | ---: | ---: |
| Complete JSON, mean | 30.062 s | 24.475 s | 18.6% |
| Complete prose, mean | 22.812 s | 19.003 s | 16.7% |
| Fixed 24-request mix, total | 634.481 s | 521.738 s | 17.8% |

Both matched start orders improve, by 16.8% and 18.7%. The source-balanced
first halves improve by 23.1%, the second halves by 11.4%. This qualifies
the earlier short comparison's 37.0% observation below: that larger gain
does not describe sustained traffic. First-token time roughly halves, but
subsequent generation is 5.7% slower for JSON and 15.4% slower for prose.
All twelve measured JSON pairs have identical text and usage. All four
starts retain the same 7/8 fidelity score and identical answers, including
the same code-trace failure. The [sustained protocol and complete record](sustained-prefill.md)
retain every output, the reference-based prose review, and limitations.

The following reader isolation holds runtime `132b629` and staged GPU
prefill fixed. Only the buffered/cached file policy changes, in
buffered/cached/cached/buffered start order, using the same sustained client:

| Measured client response | Buffered staging | Cached staging | Cached wall-time increase |
| --- | ---: | ---: | ---: |
| Complete JSON, mean | 24.320 s | 24.701 s | 1.6% |
| Complete prose, mean | 17.221 s | 19.647 s | 14.1% |
| Fixed 24-request mix, total | 498.493 s | 532.174 s | 6.8% |

Cached staging regresses in both matched orders, by 5.6% and 7.9%.
Source-balanced first halves are tied; cached staging is 15.7% slower in
the second halves. Eighteen of 24 matched responses regress, including an
aggregate 2.3% penalty among thirteen identical-text pairs. Keep buffered
GPU staging for this sustained workload. The
[reader-isolation record](sustained-reader.md) retains all checks and raw
evidence. These two experiments have different baselines; their percentages
cannot be added or treated as a direct original-versus-buffered comparison.
Both completed gates restored and verified the original system service.

## Complete agentic task wall time

The [controlled Pi comparison (#39)](https://github.com/jomcgi-org/freetoken-fork/blob/perf/4090-agentic-runtime-gate/docs/pi-agentic-runtime-gate.md)
measures the original `3a67403` runtime against the optimized `c0775ea` stack
with buffered GPU DISK prefill and mmap HOT staging. Four measured continuing
coding tasks per runtime take 1506.667 seconds (25m07s) baseline versus
1158.016 seconds (19m18s) optimized, **23.1% less task wall time**. Both matched
start orders improve, by 4.8% and 37.6%. All eight measured tasks and four
warmups pass all three independent cumulative checks; full outputs and repair
sequences were reviewed. The original service is restored and verified.

This is observed agentic throughput with model-behavior variation. Optimized
tasks use 67 calls and 22752 output tokens, versus 85 calls and 26999 tokens
baseline. The final baseline task uses 29 calls and five failed test commands
before removing an incorrect test expectation. All repair time is retained.
The unequal work and large spread between orders prevent attributing 23.1%
solely to faster inference. This result is separate from the fixed-request
17.8% sustained comparison above, and one Cache task does not establish broad
quality equivalence.

Both modes request diagnostic flags off. The baseline still calls legacy
disk/PLE status-statistics helpers; their removal is included in the optimized
stack. This gate does not isolate telemetry savings. Radix prefixes are enabled,
capacity and graph size are one, and the fixed 64K FP8 K/V pool is 0.80 GiB.
All four starts match 3753 expert slots and 2296 protected HOT rows. These
allocations differ from the earlier naive-cache experiments.

## Earlier combined short-sequence wall time

The short comparison used original `3a67403` with CPU DISK prefill against
combined `78848ce` with staged GPU prefill, published HOT reuse, and
`--moe-disk-prefill-io cached`. The real serving CLI selects the reader,
without a constructor probe. Four isolated starts use
original/optimized/optimized/original order, two full-response warmups per
start, and four measured requests per start. Native binary identities and
worker mappings are verified. Diagnostics, GPU timing, HOT persistence, and
KV reuse are off, with identical placement and cache geometry.

| Measured client response | Original | Combined | Wall-time reduction |
| --- | ---: | ---: | ---: |
| Complete JSON, mean of four requests | 36.703 s | 23.765 s | 35.3% |
| Prose at a 192-token cap, mean of four requests | 28.192 s | 17.089 s | 39.4% |
| Fixed mix, total of eight requests | 259.577 s | 163.413 s | 37.0% |

All eight matched responses improve. The first matched start pair is 32.2%
shorter; the reversed pair is 41.8% shorter. Mean first-token time falls from
15.738 to 6.029 seconds for JSON and from 15.188 to 6.169 seconds for prose.
Unlike the earlier combined CPU comparison below, both start orders favor
the optimized configuration. This measures the full configuration and does
not assign additive percentages to individual changes.

All twelve JSON responses, including warmups, finish normally and pass strict
value, integer-type, key-order, and multiplicity checks. One original response
uses extra whitespace and 449 output tokens; the other eleven use 383.
Seven measured pairs have matching usage, and that subset still takes 35.5%
less wall time (220.144 to 141.901 seconds). Three JSON pairs have identical
text. Every start scores 7/8 on the long fidelity questions, failing the same
code trace: the first original start answers `100`, the others `108`, versus
the correct `68`. Prose differs, reaches its output cap, and remains unscored.

The [complete record](../bench/results/4090-combined-cached-wall-20260905.json)
retains exact drivers and clients, source and native identities, raw outputs,
matching geometry, journals, I/O snapshots, and reproducible analysis. The
original service was restored and verified with a real completion. Retained
host page cache and limited adaptation history still constrain a general
production claim. The completed sustained result above measures a smaller
gain with longer serving-state history and complete prose responses.

## Earlier combined CPU comparison

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

The sustained combined comparison makes this visible. JSON first-token time
falls from 12.837 to 6.276 seconds, but subsequent generation grows from
17.225 to 18.199 seconds. Prose first-token time falls from 11.651 to 6.123
seconds, while subsequent generation grows from 11.161 to 12.879 seconds.
Complete-response wall time is the gate because prefill and generation move
in opposite directions.

The sustained reader isolation identifies a reader-policy regression. In its
second halves, mean whole-worker storage reads are 1.055 GiB per response
with buffered staging versus 5.679 GiB with cached staging. Those counters
do not separate prefill, decode, PLE, or adaptation. The cache-aware reader
direct-reads an entire expert row if any part is nonresident, so redundant
reads of resident bytes and reduced admission of recurring weights remain
candidates to investigate.

HOT updates expose another concrete cost. One warmup in the first cached
start stages 0.50 GiB in 5.5 seconds and triggers automatic bandwidth
backoff. For this single batch, the timer excludes the initial GPU readiness
wait and later GPU installation. It includes host copies from file-mapped
tensors under concurrent serving; it does not isolate faults, CPU scheduling,
or memory copying. The experimental buffered HOT reader in
[#35](https://github.com/jomcgi-org/freetoken-fork/pull/35) targets this work
without changing the selected experts or adding a weight buffer. Its completed
sustained comparison loses: 465.939 seconds with mmap versus 557.449 seconds
with buffered HOT reads, a 19.6% increase. Both matched orders regress and
21 of 24 responses are slower. All twelve measured JSON pairs have identical
text and usage, with a 23.6% penalty. Keep mmap HOT staging. Different
readahead controls and actual adaptation transitions are possible contributors,
not established causes; the
[HOT reader record](https://github.com/jomcgi-org/freetoken-fork/blob/perf/4090-hot-staging-io/docs/hot-staging-io.md)
retains those limits and the complete output review.

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

## Earlier fork measurements that constrain the next experiment

The fork's existing [results log](../bench/RESULTS.md) records useful negative
controls. In its September 3 and 4 rounds, moving the
WILLNEED callback after worker notification reduced wake time but increased
faults and total CPU windows. A decode-only worker cap and a spin-then-wait
barrier also failed to beat the final warm control consistently. These
reports provide no reason to repeat those policies without new evidence.

The same log reports much better HOT coverage after a production adapter has
run long enough, approximately 78-93%, than in its short starts, approximately
59-69%. Those are historical observations with different traffic and timing
settings; they are not measurements of the current benchmark. In particular,
the fixed-HOT diagnostic's 11% coverage above must not be treated as typical
of an adapted production server.

The completed sustained gates disable plan persistence and KV reuse, use
full warmups, retain automatic adaptation, and reverse server order. They
cover a finite continuous workload and do not qualify idle convergence,
very long contexts, concurrency, or production prefix reuse. The source-path
tests separately establish byte and arithmetic parity.

## Changes and evidence

PR states below describe this evidence update. Positive percentages are time
reductions unless explicitly labeled as regressions; kernel speedups and
model wall time have separate meanings.

| Change | Evidence | Delivery |
| --- | --- | --- |
| Selective short-prefill transfers and diagnostic gating | About 15% lower 76-token TTFT in the initial acceptance run; larger prompts were mixed | [#27](https://github.com/jomcgi-org/freetoken-fork/pull/27), ready |
| Parallel HOT route preparation | 20,480-route kernel: 1,719.7 to 39.3 microseconds; model-wide gain remains unquantified | [#28](https://github.com/jomcgi-org/freetoken-fork/pull/28), ready |
| Concurrent CPU/GPU prefill | Ordering and placement confounded the early model gain | [#29](https://github.com/jomcgi-org/freetoken-fork/pull/29), draft |
| Reuse CPU input quantization across routes | Whole-request reductions of 5.0%, 11.4%, 16.6% at 76, 524, 2,060 prompt tokens; all 48 prefill pairs faster and all 56 timing pairs identical in text and usage | [#30](https://github.com/jomcgi-org/freetoken-fork/pull/30), ready |
| Gate diagnostic-only decode classification and idle history readback | Removes ordinary decode's reduction/counter work and idle coverage's GPU-to-CPU history synchronization; 99 focused Linux tests pass, with no independent wall-time claim | [#31](https://github.com/jomcgi-org/freetoken-fork/pull/31), ready; also included in #33 |
| Correct temporary materialization ownership | Prevents sparse scratch from advertising uncopied experts as hits; protected HOT bytes retained | [#32](https://github.com/jomcgi-org/freetoken-fork/pull/32), ready |
| Selected DISK staging and exact HOT VRAM reuse | Earlier direct CPU comparison: 31.6% lower prose wall time, mixed JSON gains; included in the sustained combined 17.8% result | [#33](https://github.com/jomcgi-org/freetoken-fork/pull/33), draft |
| Cache-aware staged reads | Short reader gate: 18.3% less wall time; sustained isolation: 6.8% slower overall and 15.7% slower in second halves; 99 CUDA checks pass normally and under memcheck | [#34](https://github.com/jomcgi-org/freetoken-fork/pull/34), draft; keep buffered staging for the sustained workload |
| Buffered reads into HOT staging rows | Sustained wall time is 19.6% longer than mmap, with both orders slower; 164 focused Linux/CUDA checks and 25 zero-error memcheck checks pass | [#35](https://github.com/jomcgi-org/freetoken-fork/pull/35), draft; keep mmap HOT staging |
| Concurrent complete-response client and gate | Fixed-capacity 1/4/4/1 starts: 498.127 to 512.904 seconds, 3.0% longer overall with conflicting orders; mean individual latency 20.755 to 82.122 seconds; 18 client checks pass | [#36](https://github.com/jomcgi-org/freetoken-fork/pull/36), ready as benchmark tooling; no dependable batching gain |
| Reuse weight unpacking across shared decode routes | At batch four and ten active routes, resident-weight native task time falls 21.2% with five shared experts and 43.8% with ten; 69 Linux/CUDA checks and 352 bitwise-equal timing pairs pass; model wall time pending | [#37](https://github.com/jomcgi-org/freetoken-fork/pull/37), draft; defaults off |
| Multi-turn Pi task benchmark | All three independent task stages pass in a live integration, 18 model calls and two recovered test failures; 18 focused client checks pass; completed runtime comparison is reported separately in #39 | [#38](https://github.com/jomcgi-org/freetoken-fork/pull/38), ready as benchmark tooling |
| Complete Pi runtime comparison | Four measured tasks per mode: 1506.667 to 1158.016 seconds, 23.1% less wall time; both orders improve, all checks pass, but generated work differs; 9 controller and 15 summary checks pass | [#39](https://github.com/jomcgi-org/freetoken-fork/pull/39), ready as benchmark and evidence |
| Skip resident CPU populate reads | Eleven Linux correctness checks pass; component population plus mapped-byte consumption improves 15.0% warm and 2.8% mixed, regresses 1.0% cold; full-model comparison running, no inference gain yet | [#40](https://github.com/jomcgi-org/freetoken-fork/pull/40), draft; defaults off |

The main dependency chain is #27, #28, #30, #32, #33. #31 is independently
reviewable from #28 and is cherry-picked into #33. #29 remains separate.
#34 evaluates a DISK prefill reader on top of #33; #35 evaluates HOT staging
on top of #34, #36 adds a concurrency client on top of #35, and #37 adds the
native decode experiment on top of #36. The original
serving checkout is restored and verified after
every completed model gate; these changes are delivered in PRs.

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
including the same wrong code-trace answer. The earlier combined CPU comparison
also retained 7/8 but changed that incorrect answer. Prose from the earlier
192-token-cap probes remains unscored. The three sustained comparisons generate
complete prose with a 1,024-token ceiling. Their artifacts include an
assistant review of all responses against the seven-constraint reference
specification. Both modes have omissions or unsupported explanatory
connections. These small audits do not establish broad model-quality
equivalence or statistical noninferiority.

Use `--moe-collect-stats` and `--moe-step-timing` only for diagnostic runs.
Functional HOT histories, session-prefetch observations, and the WILLNEED
fault guard still run when diagnostics are off. The selected-union host
readback is required transport work. Optional `/proc` I/O snapshots run
outside the client timer when the new `--phase-io` diagnostic is off, as it
is in both sustained gates. That diagnostic adds a snapshot at first text,
labels its rows, and includes observation cost in wall time. Functional
staging timers still drive automatic backoff; the ordinary runtime is not
free of all timing calls. Qualify performance with complete responses,
matching placement and token counts, full warmups, and reversed request
orders. The staged wall measurements precede the final #31 cherry-pick;
the combined comparison includes it. No independent throughput percentage
is claimed for that gate.

## Next performance gate

Keep CPU DISK prefill as the serving default while broader qualification
remains open. For the measured sustained workload, the preferred opt-in is
`--moe-disk-prefill staged --moe-disk-prefill-io buffered`. The cache-aware
reader's short 18.3% gain failed to carry into sustained use. The
[direct-only negative control](direct-prefill-io.md#whole-response-result-direct-only-reads-do-not-qualify)
was also slower overall and increased worker storage reads to about 27 GiB
per response. These results favor preserving useful RAM residency over
unconditional cache bypass on this machine.

The HOT staging comparison at `878d723` also favors the existing mmap path.
Do not infer a serving gain from a cheaper host-copy operation. Matching
descriptor advice remains a possible follow-up, with no qualified gain yet.

The completed concurrent request gate uses the preferred buffered GPU DISK
prefill and mmap HOT staging. The
[#36 client](https://github.com/jomcgi-org/freetoken-fork/blob/perf/4090-concurrent-wall-client/docs/concurrent-wall.md)
preserves the same scored prompts and complete outputs, measures the entire
workload's elapsed time, and reports individual response latency separately.
With server capacity four fixed, twenty-four measured responses take 498.127
seconds at concurrency one and 512.904 seconds at concurrency four. The 3.0%
aggregate increase hides conflicting matched orders: four requests are 18.7%
slower in one and 12.1% faster in the other. Mean individual latency increases
from 20.755 to 82.122 seconds. All measured JSON pairs have identical text and
usage, and every start retains identical fidelity answers and 7/8. The complete
prose review includes omissions and unsupported connections in both modes.
This finite comparison does not establish a dependable batching gain.

Storage and adaptation remain material context. The two concurrent starts
read 63.630 and 30.286 GiB during measurement, and the final single-request
start backs off its HOT interval from 1,000 to 2,000. Those are whole-worker
counters and actual controller transitions, not isolated causal measurements.
All four starts have equal geometry, including 3,920 expert slots and 1,024 KV
pages; the fixed four-request allocation differs from earlier single-capacity
comparisons. Do not combine their percentages.

A separate pending gate is complete model wall time for
[#37's native decode weight reuse](https://github.com/jomcgi-org/freetoken-fork/blob/perf/4090-decode-weight-reuse/docs/decode-weight-reuse.md).
The existing CPU schedule groups routes by expert but repeats weight unpacking
inside each group. The opt-in AVX-512 VNNI path shares that work across up to four
routes while retaining the serial decode accumulators, activation handling, and
original top-k reduction. All 69 focused Linux/CUDA checks pass, including direct
FP32-bit comparisons and both captured CPU/GPU transports. All 352 paired native
timing outputs match bitwise. At batch four with ten active routes, native task
time falls 21.2% with five shared experts and 43.8% with ten, while no sharing is
effectively flat. These synthetic resident-weight results do not establish model
throughput. Keep it off by default until a non-debug complete-response gate
measures the same runtime and binary with reuse off/on in both start orders.

The tested automatic decode-history preference was reverted: it improved
JSON by 26.8% but slowed prose by 8.6%. That experiment changed only placement
planning, yet still failed the general throughput gate. It provides no reason
to bias the model toward HOT experts.

The multi-turn Pi tooling in #38 is now qualified by the completed #39
comparison above. It retains conversation history, local tests, follow-up
requirements, and independent cumulative checks within each task clock.
The 18 client checks, nine controller checks, and fifteen summary checks pass
on Linux and macOS. Systemd EOF and heartbeat-timeout recovery probes passed,
and the completed run accounts for all 229 expected HTTP completions without
inference errors. The full record preserves warmups, failed local commands,
final files and verified restoration.

The next single-agent experiment is
[#40's optional resident CPU population check](https://github.com/jomcgi-org/freetoken-fork/blob/perf/4090-populate-resident/docs/resident-populate.md).
CPU prefill currently copies selected file bytes into reusable scratch, discards
them, and computes from original bank mappings. Owned-file residency hints can
skip those copies when all covering pages are in RAM, while uncertain hints
retain buffered reads. This preserves routing and packed weight bytes. All
11 focused Linux checks pass without skips, including real warm-file skipping
and unchanged mapped bytes. The completed private-file component comparison
includes population and later checksum consumption of mapped bytes. Wall time
improves 15.0% warm and 2.8% mixed, and regresses 1.0% cold, with all checksums
matching. These are component effects. The full-model off/on/on/off comparison
is running on one frozen runtime with mixed short and long prefill, diagnostic
flags off and identical memory geometry. Its first startup failed before any
timed response because a native PLE dependency was missing; that failure and
verified service recovery are retained. The repaired run is detached under
systemd and must finish both orders, output review and recovery before an
inference gain can qualify. The option remains disabled. The pending native
decode reuse experiment targets shared experts across multiple routes and does
not accelerate the single-route case by itself.

The [LayerScale and continuation-state review](layerscale-4090-review.md) identifies
a separate agentic opportunity. FreeToken already preserves hybrid state with
prefix caching, but unaligned response endings may lose the latest recurrent
snapshot. Existing Pi usage records frequently resume at the preceding prefill
boundary. Exact token tracing must distinguish a missing snapshot from a tool
call that the client reserialized before changing reuse policy. Prompt-lookup
drafting is also a candidate that needs no extra draft-model weights; neither
idea has a qualified 4090 speedup yet.

A [source review of the existing MTP graph branch](mtp-4090-review.md) identifies
prerequisites for a separate sequential-decode experiment. The deployed FTW index
contains no MTP tensors. The branch also records six CUDA timing events per
verification step without a diagnostic guard, while host PLE staging, state
snapshots, rejected-seed replay, and draft-head VRAM compete with any verification
gain. Gate the timing instrumentation and retain target-equivalence checks before
qualifying MTP. The completed Pi comparison kept MTP off.
