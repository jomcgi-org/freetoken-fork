# Real-source prefill depth comparison

`bench/prefill-depth.py` prepares one fixed manifest and measures cold prefill,
an immediate cached repeat, and a separate decode request after each document.
It addresses the measurement portion of fork issue #80. It does not change the
serving defaults or qualify a new profile by itself.

Prepare once on the serving machine, before timing any arm:

```sh
python bench/prefill-depth.py prepare \
  --source . --tokenizer /path/to/flash-e2m1.ftw \
  --depths 8000 32000 100000 --runs 3 --output manifest.json
```

The manifest contains excerpts from the selected source tree and unique prefixes
per depth and repetition. Its rendered inputs are within 256 tokens below the
requested depths. All arms must use the same manifest. The model must support
the largest input plus the 192-token output allowance; the node-4 comparison
keeps its existing 100352-token KV reservation.

Start an arm with the qualified launch script and override only the chunk size
being tested. For the cold sweep, disable disk prefix persistence in **every**
arm with `--kv-disk-cache-gib 0`, retain the ordinary radix cache, and restart
between arms. This means cold **prefix** state, not flushed operating-system
file caches. Wait for `/health` JSON `status` to become `ok`; HTTP 200 alone also
occurs during model loading.

```sh
python bench/prefill-depth.py measure \
  --manifest manifest.json --arm chunk-2048 \
  --base-url http://127.0.0.1:18090 --output chunk-2048.jsonl
```

Run 2048, 4096 and 8192, then repeat the control or reverse the order. Record the
exact source revision, native extension, model, launch command, startup memory
budget and layer residency beside each result. The native CPU task descriptor
accepts up to 2147483647 tokens, so it does not bind these candidate chunk sizes;
actual host and device allocations do.

Each result retains the complete streamed response, token usage, finish reason,
request hash, first-generated-text latency, total wall time and answer check.
The decode-rate estimate excludes latency before the first generated text and
uses completion tokens minus one divided by the interval to the last text
chunk. It is a client-observed estimate, not an engine kernel timer. Role and
keepalive frames do not end the first-token wait. The repeat measures immediate
in-memory reuse; it does not establish disk-restore or tool-continuation speed.

Reject truncated, failed or incorrectly copied answers. Check `cold_valid` and
actual cached-token counts before accepting a cold measurement. Compare request
hashes and output counts across arms, retaining differences rather than silently
discarding slow or failed runs. JSON copying is a narrow fidelity check, not a
general quality gate. Chunk changes can change floating-point rounding and
generated text. A finalist still needs the existing matched multi-turn/coding
checks, a disk-prefix restore check, and repeated measurements at long context.

Do not change the default until the harness-root persistence gap in #81 is
addressed or its effect on the chosen profile has been explicitly qualified.
Larger chunks can cause more system roots to fall inside a final chunk, where
the existing implementation does not persist them independently.

## Initial node-4 screening, 2026-09-22

Runtime `4fbc4ebf3f7d0c4039893b0471faf231c7a15ba0`, RTX 4090, 61.91 GiB host
RAM, 100352 reserved KV tokens, 14 CPU threads and 6 GiB protected HOT budget.
All arms kept 20 PINNED and 28 DISK layers, 82 protected rows per DISK layer,
3589 expert slots, and 2.12 GiB free GPU memory after graph capture. The CPU
extension SHA-256 was
`c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f`.

The order was 2048, 8192, 4096, 2048. Each server started from its static expert
profile with empty prefix state. Each arm ran an approximately 8k source prompt,
its cached repeat and a separate decode request, then the same sequence at 32k.
Actual input counts were 7959 and 31958. This is one fixture per depth and two
control observations, not a confidence interval or a selected serving default.

| Chunk size / order | Cold TTFT, 8k | Cold TTFT, 32k | Decode during cold 32k answer | Independent decode after 32k |
| --- | ---: | ---: | ---: | ---: |
| 2048 / first | 46.61 s | 82.38 s | 11.87 tok/s | 29.16 tok/s |
| 8192 / second | 12.86 s | 52.30 s | 5.30 tok/s | 29.52 tok/s |
| 4096 / third | 20.62 s | 70.26 s | 10.12 tok/s | 31.09 tok/s |
| 2048 / last | 33.61 s | 96.37 s | 10.21 tok/s | 29.12 tok/s |

All 24 responses passed the JSON-copy checks. The cold and repeat answers each
used 81 output tokens; independent decode used 451. All cold requests reported
zero cached tokens. The 8192 cold 32k request completed in 67.91 s versus
89.22 and 104.32 s for the controls, but its immediate decode and cached repeat
were slower: repeat wall was 4.70 s versus 3.63 and 3.68 s. Independent subsequent
decode recovered to the control rate. That transient cost remains part of the
qualification, not a discarded measurement.

The governor charged prefill scratch of 0.37, 0.64 and 1.18 GiB for chunks 2048,
4096 and 8192 respectively. Expert layer placement was unchanged and no host
pressure warning appeared in the saved arm journals. These startup estimates
do not substitute for the pending 100k pressure measurement.

Private full-response artifacts, manifests, launch commands and journals are in
`node-4:/var/lib/longhorn/nvme-02/freetoken/results/prefill-depth-20260922/`,
using `screen3-*.jsonl`. Earlier controller setup failures are preserved under
different names and excluded from this table. The original serving service was
restored successfully after the sweep.

## 100k screening and host pressure

The same runtime then ran one fixed 99,959-token prompt with 81 output tokens,
its immediate repeat, and a separate 451-token decode request. The order was
8192, 8192 with `--moe-hot-adapt-prefill-run-cap-frac 0.1`, then 2048. The cap
limits expert-cache swaps during a prefill run to a fraction of the HOT budget;
zero disables this additional cap. All other profile settings stayed fixed,
including disabled disk-prefix persistence and enabled in-memory radix reuse.

| Chunk / prefill swap cap | Cold TTFT | Cold wall | Cold answer decode | Repeat wall | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 / disabled | 167.77 s | 175.43 s | 10.71 tok/s | 8.36 s | 25.13 tok/s |
| 8192 / 0.1 | 141.73 s | 147.38 s | 14.65 tok/s | 8.70 s | 24.03 tok/s |
| 2048 / disabled | 267.03 s | 272.32 s | 15.77 tok/s | 6.30 s | 26.09 tok/s |

All nine responses passed, with identical request hashes, answer bytes and output
counts across arms for each phase. Cold requests had zero cached tokens; repeats
had 99,904. Repeat TTFT was 2.67, 2.68 and 2.72 seconds respectively, so the
repeat regression was after first text. Repeat decode estimates were 14.38,
13.54 and 23.14 tok/s. The capped candidate improved cold TTFT by 1.88x, but its
38% longer repeat wall and lower subsequent decode rate prevent a claim that it
preserves decode performance. These are single observations, not a qualified
default or a confidence interval.

Two-second samples of `/proc/meminfo`, `/proc/pressure`, `/proc/vmstat` and
`/proc/diskstats` were correlated with each cold request's client interval:

| Chunk / cap | Sampled / request seconds | Min available RAM | Memory full stall | I/O full stall | Main NVMe reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 / disabled | 164.4 / 175.4 | 28.65 GiB | 14.17% | 26.69% | 76.86 GiB |
| 8192 / 0.1 | 144.2 / 147.4 | 28.47 GiB | 9.43% | 21.05% | 59.44 GiB |
| 2048 / disabled | 270.5 / 272.3 | 29.47 GiB | 2.20% | 10.85% | 51.25 GiB |

Stall percentages use deltas of the kernel's cumulative `full` pressure counters,
not averages of its rolling averages. NVMe reads use sector-counter deltas for
`nvme1n1`. These are whole-host counters, not process-exclusive attribution. The
monitor started after the first request began, and two-second sampling also
omits interval edges. Available memory alone did not capture the pressure:
larger chunks were faster despite more sampled disk traffic and stall time.
This motivates the host-budget diagnosis in #82, but does not prove a specific
memory-governor fix.

Artifacts use `long1-*-chunk-*.jsonl`, corresponding command/journal files,
`long1-host-pressure.jsonl` and `summarize-long-pressure.py` in the same private
results directory. The continuation and prefill-to-decode transition comparisons
below extend this screening. The harness-root
restart check also exposed a separate tokenizer-template rejection of system-only
messages; PR #85 addresses that alongside final-chunk snapshots. Its eager-restore
restart checks subsequently passed at both 2048 and 8192 tokens, as documented
in `docs/disk-prefix-cache.md` on that branch. No serving default has been changed.

## Matched continuation screening

The existing `fixed-continuation-wall.py` protocol ran three conversations of
three turns each, first on 2048 chunks and then on 8192 with the 0.1 prefill swap
cap. Each arm restarted from the same qualified runtime with disk-prefix storage
disabled. Conversation 1 was the prescribed warm-up; conversations 2 and 3 were
measured. Initial prompts were about 2k tokens, and answers about 448 tokens.

| Chunk / cap | Warm-up conversation | Measured conversation 2 | Measured conversation 3 | Measured mean |
| --- | ---: | ---: | ---: | ---: |
| 2048 / disabled | 98.25 s | 68.63 s | 64.48 s | 66.55 s |
| 8192 / 0.1 | 91.42 s | 67.25 s | 64.13 s | 65.69 s |

All 18 responses passed. The protocol's `fixed_work_mismatches` check found no
differences in request bodies, answer messages, finish reasons, prompt counts or
output counts. The candidate reused 2048 tokens on turn 2 versus 1984 for the
control; both reused 2944 on turn 3. The measured mean improved by only 1.3%,
while the combined measured follow-up turns were about 1.6% slower. This small
sample is broadly similar total wall time, not proof of a general agent-quality
or decode-speed improvement, and it does not remove the 100k repeat regression.
Full results are under `cont1-*-chunk-*/session-*/result.json` beside the exact
launch commands and journals in the private results directory.

## Prefill-to-decode transition screening

Three additional 8192-token arms used the same 100k manifest and baseline runtime.
They tested `--moe-hot-adapt-post-prefill-tick on` with prefill swap caps of 0.1
and 0.01, followed by cap 0.01 with the tick off. The extra tick starts adaptation
toward decode after prefill; it does not change the model or allocate a larger
protected expert cache.

| Prefill cap / post-prefill tick | Cold TTFT | Cold wall | Cold answer decode | Repeat wall | Repeat decode | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 / on | 138.06 s | 148.08 s | 8.13 tok/s | 6.83 s | 21.35 tok/s | 24.73 tok/s |
| 0.01 / on | 145.51 s | 154.06 s | 9.56 tok/s | 7.16 s | 19.53 tok/s | 24.26 tok/s |
| 0.01 / off | 133.40 s | 142.00 s | 9.48 tok/s | 7.45 s | 17.67 tok/s | 23.90 tok/s |

Across all six 100k arms, all 18 responses passed and each phase had identical
request hashes, answer bytes, input counts and output counts. Prefix-hit counts
also stayed fixed at zero for cold requests and 99,904 for repeats. The tick
improved repeat decode relative to the 0.1-cap/no-tick arm, but slowed decode
during the cold answer. The smallest cap without the tick had the fastest cold
TTFT, almost 2x the 2048 control, while its repeat remained 18% longer and its
independent decode rate about 8% lower. No tested configuration demonstrated
both the cold-prefill gain and preserved decode across these measurements.

Keep the serving default at 2048. The binding qualification issue in this sweep
is cached/decode performance, not an observed allocation failure. Larger chunks
also showed higher host pressure; a successful allocation is not sufficient
evidence for selecting them. The next work is to explain and validate the
prefill-to-decode cache transition, repeat comparisons in reversed order, and
qualify harness-root reuse with the selected lazy-restore configuration. The experiments establish a
promising prefill opportunity, not a new qualified serving profile.

Additional response artifacts use `long2-posttick-*.jsonl` and
`long3-cap001off-*.jsonl`, with launch commands, journals and pressure samples
beside them. The benchmark harness passed six targeted tests on both the local
development machine and node-4 Linux. The screening covered 42 depth-benchmark
responses plus 18 continuation responses; all passed their narrow fidelity
checks. Controllers completed successfully and restored the original service
configuration after each comparison.

## Adaptation-clock follow-up

Saved journals exposed a clock-phase difference: the 2048 control first reranked
for decode at routed token 99,992, while the 8192 arms without a forced tick
waited until 100,134, during the cached repeat. The automatic fill interval is
166 tokens. Its first chunk consumes ticks through 1992 or 8134 respectively;
switching to the steady 1000-token interval retains that offset in
`HotAdaptTokenClock.set_interval`. Both forced-tick arms additionally logged
bandwidth back-off from interval 1000 to 2000 after their first decode tick.

A fresh comparison tested the original 2048/automatic control followed by 8192
with cap 0.1 and explicit `--moe-hot-adapt-interval-steps 1000`. The candidate's
first decode rerank moved to token 100,000 as predicted, with no automatic
back-off. Model, native extension, prompt and cache configuration stayed fixed.

| Configuration | Cold TTFT | Cold wall | Cold answer decode | Repeat wall | Repeat decode | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 / auto / uncapped | 265.54 s | 270.82 s | 15.66 tok/s | 5.57 s | 21.31 tok/s | 24.38 tok/s |
| 8192 / fixed 1000 / cap 0.1 | 149.77 s | 160.43 s | 7.65 tok/s | 6.31 s | 22.33 tok/s | 22.82 tok/s |

All six additional responses passed and matched across arms in request hashes,
answer bytes and input/output counts. Repeat hits remained 99,904 tokens. The
candidate's repeat decode improved relative to the earlier automatic 8192/0.1
arm, consistent with the timing hypothesis. Its repeat wall was still 13% longer
than the fresh control, cold-answer decode was slower, and independent decode
was about 6% slower. Clock alignment alone did not qualify the faster profile;
the earlier and later observations also show why one control is insufficient.
Retain 2048 pending a validated solution to the remaining transition costs.

Artifacts use `clock1-*-chunk-*.jsonl`, matching command/journal files and
`clock1-host-pressure.jsonl` in the same results directory. Including this
follow-up, 48 depth responses and 18 continuation responses passed their narrow
checks. No serving default was changed.

## Uncapped fixed-interval follow-up

A reversed-order comparison on the same runtime tested 8192-token chunks with
fixed interval 1000 and no prefill swap cap, then a fresh 2048/automatic control.
This separates the earlier fixed-interval candidate from its restrictive 0.1
prefill swap cap. All other settings and the 100k manifest remained unchanged.

| Configuration | Cold TTFT | Cold wall | Cold answer decode | Repeat wall | Repeat decode | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 / fixed 1000 / uncapped | 151.28 s | 160.22 s | 9.15 tok/s | 6.45 s | 23.18 tok/s | 25.92 tok/s |
| 2048 / auto / uncapped | 268.31 s | 274.81 s | 12.72 tok/s | 6.09 s | 24.11 tok/s | 25.98 tok/s |

All six responses passed. Comparisons performed on node-4 confirmed identical
request hashes, complete answers, prompt counts and output counts for each
phase. Cold hits remained zero and repeats reused 99,904 tokens. Independent
request wall time was 22.09 versus 21.99 seconds. The candidate substantially
reduced cold prefill and preserved independent decode in this sample, but its
cold-answer decode remained slower and its repeat wall was 5.9% longer.
This is not yet an unqualified replacement for the selected profile.

The candidate's first decode adaptation tick reported 36.63% protected-expert
pair coverage, similar to the earlier automatic control's 36.53%, rather than
the capped fixed-interval candidate's 23.74%. Placement alone therefore does
not explain the remaining cold-answer penalty. These counters do not isolate
host page-cache effects or adaptation work overlapping the transition.

Artifacts use `clock2-*.jsonl` and matching commands/journals in the same private
directory. The controller completed successfully and restored the original
configuration. Including this follow-up, 54 depth responses passed the narrow
fidelity checks; no larger chunk default was selected.


## Deferred CPU workspace experiment

Staged GPU prefill constructs a CPU MoE executor for decode and fallback, but
previously allocated the native CPU-prefill batch workspace immediately. That
workspace scales with maximum chunk size even if staged prefill never calls the
CPU batch path. The staged profile now defers native batch setup until its first
actual CPU-prefill call. Ordinary CPU prefill keeps eager setup. The startup log
reports `deferred` with zero allocated batch bytes until that first call.

The host-memory governor still charges the full possible workspace. This keeps
expert placement and the fallback memory allowance unchanged. Setup is attempted
once; missing kernels or allocation failure retain the serial fallback without
repeated allocation attempts. The workspace remains allocated if CPU fallback
is used. Its capacity is now bounded by the smaller of the scheduler chunk and
one below the staged crossover (1023 rows at the default 1024-token threshold).
Chunks at or above the threshold stage on the GPU. This preserves the saving
after short continuations without releasing and reallocating buffers. A threshold
of one retains the native API's minimum one-row capacity, allocated only if used.
The governor deliberately retains its conservative full-chunk allowance.

The initial lazy-allocation revision passed 40 targeted Linux tests with no skips.
The bounded-capacity follow-up passed all 75 targeted Linux tests with no skips.
Matched node-4 performance measurements remain pending. Tests
cover zero initial native batch bytes in lazy mode, allocation on first prefill,
serial-reference numerical parity, repeat-buffer reuse, and one-time setup
failure handling. No faster serving profile has been qualified by this change.


### Bounded workspace screening

Three fresh runs compared revision `644dcc1` at 4096 and 2048 tokens against
selected revision `11654ed` at 2048. All used automatic HOT cadence, no prefill
swap cap, the default host reserve, disabled disk prefix caching, and the same
continuation schedule followed by the frozen 100k cold/repeat/post-decode case.

| Metric | Bounded 4096 | Bounded 2048 | Selected 2048 |
| --- | ---: | ---: | ---: |
| Measured continuation mean | 65.44 s | 68.11 s | 69.89 s |
| Cold TTFT | 172.84 s | 262.66 s | 261.36 s |
| Cold wall | 176.33 s | 266.16 s | 265.05 s |
| Cold answer decode | 23.77 tok/s | 23.81 tok/s | 22.34 tok/s |
| Repeat TTFT | 5.06 s | 2.57 s | 1.53 s |
| Repeat wall | 7.87 s | 5.58 s | 4.56 s |
| Repeat decode | 29.41 tok/s | 27.37 tok/s | 27.19 tok/s |
| Subsequent decode | 27.36 tok/s | 31.04 tok/s | 26.90 tok/s |

All 27 continuation calls and nine depth responses passed with exact request,
answer, and token-count parity across profiles. Cold hits were zero; repeat hits
were 99,904 tokens. The controller restored the selected service and its health
check passed. Candidate logs confirmed deferred 1023-row capacity with no CPU
batch degradation warnings. Native libraries were identical across profiles.

The larger candidate reduced cold TTFT by 34% and preserved the measured decode
rates against this fresh baseline, but repeat wall regressed by 73%. The smaller
candidate also regressed repeat wall by 22%. Neither result qualifies a serving
change from this sample. Continuation usage includes short CPU-prefill inputs
before the depth phase, so first CPU allocation during the repeat is not a
supported explanation. Compare repeated 8k/32k/100k cases before choosing a
profile or attributing the repeat delay to the workspace change.

Host-wide pressure samples covered at least 95% of cold requests and 80% of
subsequent-decode requests. The 4096 candidate had higher memory and I/O stall
percentages than the 2048 candidate in both phases, without greater total reads.
Against the selected baseline, cold pressure/read metrics were higher; after
prefill only memory stall percentage was higher, while I/O stalls and reads were
not. Both candidate post-decode windows included swap-ins but no swap-outs or
reclaim scans. These host-wide observations do not identify the affected process
or establish causality. Detailed counters remain on node-4.

Artifacts are `lazybench1-*-chunk-*` and `lazybench1-host-pressure.jsonl` under
the existing results directory. PR performance qualification remains incomplete.
