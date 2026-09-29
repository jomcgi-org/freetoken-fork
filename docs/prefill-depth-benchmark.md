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
The bounded-capacity follow-up is pending Linux tests and matched node-4
measurements. Tests
cover zero initial native batch bytes in lazy mode, allocation on first prefill,
serial-reference numerical parity, repeat-buffer reuse, and one-time setup
failure handling. No faster serving profile has been qualified by this change.

## Matched chunk curve on the deferred-workspace candidate, 2026-09-22

The bounded-capacity candidate (644dcc1) passed its 75 targeted Linux tests with
no skips, then ran on node-4 against a fresh 11654ed control. Four arms shared
one manifest: three repetitions each at 8k, 32k and 100k prompt depth, plus the
three-session fixed continuation workload before every depth measurement.
Order: control 2048, candidate 2048, candidate 4096, candidate 8192. Every arm
started with an empty disk prefix cache and the automatic HOT adaptation
interval. All 108 depth responses and 12 continuation responses passed, with
exact text, reasoning, finish-reason and usage parity across arms.

| Arm | 8k cold TTFT | 32k cold TTFT | 100k cold TTFT | 100k repeat wall | 100k post-prefill decode | Continuation walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control 2048 | 23.45 s | 72.49 s | 222.34 s | 4.25 s | 27.10 tok/s | 70.9 / 64.5 s |
| candidate 2048 | 25.01 s | 72.80 s | 226.09 s | 4.33 s | 20.88 tok/s | 78.6 / 65.5 s |
| candidate 4096 | 16.50 s | 44.28 s | 135.40 s | 4.73 s | 27.56 tok/s | 77.6 / 67.5 s |
| candidate 8192 | 9.95 s | 33.41 s | 88.64 s | 4.31 s | 26.44 tok/s | 69.7 / 67.9 s |

Values are means of three runs. The candidate at 2048 matched the control,
which is the expected result for a change that only defers and bounds CPU
scratch. Cold TTFT fell 2.5x at every depth with 8192-token chunks, and the
earlier cached-repeat penalty did not reproduce: 8192 repeat walls were 4.06,
3.78 and 4.31 seconds at 8k, 32k and 100k against 3.69, 3.67 and 4.25 for the
control. The 4096 arm's 8k repeat mean of 7.40 seconds came from a single
stalled run (14.95 s, 6.4 tok/s); its other two runs matched the control.
Post-prefill decode at 100k was within run-to-run noise of the control for
4096 and 8192; the candidate-2048 value of 20.88 tok/s came from one 100k run
at 16 tok/s and is not attributable to the code change, since the same code at
larger chunks did not show it.

Artifacts use `workspacecurve1-*` in the private results directory, with
`workspacecurve1-host-pressure.jsonl` sampled alongside. This is the first
measurement in which a larger chunk improved cold prefill without a matched
cached-decode or continuation regression.

## Larger chunks and the governor's scratch charge, 2026-09-26

A follow-up on the same candidate (644dcc1) tried 16384-token chunks with the
same manifest. The arm started healthy with the usual 2.12 GiB of free GPU
memory, but its first cold 8k prompt took 13.59 s against 9.95 s at 8192, and
the three-session continuation workload prefilled 2k-token chunks three to five
times slower than at 2048 or 8192. The sweep was stopped after the first 8k
case rather than continuing to 32768.

The startup log explains it. The host-memory governor still charged CPU prefill
scratch for the full scheduler chunk: 2.27 GiB at 16384 against 1.18 GiB at
8192. That lowered the derived pinned-bank budget from 26.70 to 25.89 GiB, which
fit 19 GPU prefill layers instead of 20 and mapped 29 MoE layers to DISK instead
of 28, with 80 rather than 82 protected HOT experts per layer. Every chunk then
staged one more layer's experts from host memory. The staged executor never
allocates that scratch: its workspace is bounded one row below the staged
crossover (1023 rows), so the charge was for memory that could not be used.

The governor now charges the same bounded workspace the executor can allocate,
through a helper shared by both. The charge no longer grows with chunk size in
staged mode, and ordinary CPU prefill keeps the full-chunk charge. Targeted
Linux tests (76, including a new chunk-independence check) passed with no
skips. Artifacts use `workspacecurve2-*`; the fixed tree is measured under
`workspacecurve3-*` at 16384 and 8192.

## Fixed-governor sweep and promotion, 2026-09-26

With the bounded governor charge, a 16384-token arm started with the same
placement as 8192: 0.23 GiB scratch, 26.77 GiB pinned budget, 20 GPU prefill
layers, 28 DISK layers and 82 protected HOT experts. Its three 8k cold requests
took 14.41, 11.78 and 10.13 s, converging on the 8192 figure as the page cache
warmed, so the per-chunk cost is not worse at 16384. The first 32k request then
failed: the scheduler reported a CUDA out-of-memory while prefilling the first
16384-token chunk with 64 MiB of GPU memory free and 22.9 GiB allocated by
PyTorch. The server recovered and rejected the request with
`server_out_of_memory`; the sweep continued to its 8192 control. Peak GPU
memory sampled through that 8192 arm was 23.67 GiB of 24.56, so 12288 is not
expected to fit either. On this 24 GiB card, 8192 is the largest chunk that
completes the 100k workload.

The 8192 control on the fixed tree passed all 27 depth responses with exact
parity against the 2048 control and the earlier 8192 arm.

| Arm | 8k cold TTFT | 32k cold TTFT | 100k cold TTFT | 100k repeat wall | Continuation walls |
| --- | ---: | ---: | ---: | ---: | ---: |
| control 2048 (2026-09-22) | 23.45 s | 72.49 s | 222.34 s | 4.25 s | 86.2 / 70.9 / 64.5 s |
| candidate 8192 (2026-09-22) | 9.95 s | 33.41 s | 88.64 s | 4.31 s | 88.0 / 69.7 / 67.9 s |
| fixed 8192 (2026-09-26) | 12.97 s | 32.25 s | 89.64 s | 6.14 s | 104.0 / 86.1 / 78.1 s |

Cold prefill matched the earlier 8192 arm at 32k and 100k; the 8k mean carries
the post-restart warm-up of its first run. Decode-heavy phases were slower than
on 2026-09-22 in this arm, and the same slowdown appeared in the deployment
checks below, so it is treated as host state rather than a code effect: the
fixed tree's 16384 arm, run minutes earlier, completed its continuation
sessions in 101.9, 71.8 and 69.7 s.

### Host memory attribution

The serving cgroup, not unrelated host activity, is what swaps. Before the
sweeps the host had no swap in use. During the 26-minute fixed-governor sweep
the host swapped out 1,035,684 pages and swapped in 168,733, with memory-stall
time of 1.0% and IO-stall time of 6.6%. Afterwards `freetoken-serve`'s cgroup
reported `memory.current` of 55.9 GiB and `memory.swap.current` of 1.15 GiB on
a 61.9 GiB host, with 28.1 GiB shmem (the pinned expert banks) and 25.8 GiB of
file-backed bank pages, and only 1.7 GiB anonymous. The swapped pages are the
part of the process the kernel may evict once the pinned banks and the pager's
file pages fill memory. The zone-normal free list was also heavily fragmented
(141,509 movable allocation stalls, 43,436 compaction stalls cumulative). This
explains why larger chunks previously showed more stalls without reading more
data, and it points at reserve sizing and swap avoidance, not chunking, as the
next decode lever.

### Promotion

The fixed tree (branch `perf/prefill-chunk-finalist-20260926`) was deployed to
`freetoken-serve` with `--max-extend-length 8192`, the disk prefix cache on,
and the previous drop-in saved for rollback. Verification on the live server
passed a short completion, 8k and 32k cold requests with correct JSON answers,
their cached repeats (7,936 and 31,936 reused tokens) and a decode check.
Cold TTFT right after restart was 13.06 s at 8k and 57.54 s at 32k, the usual
warm-up; repeat TTFT was 1.57 and 1.38 s. The three-session continuation
workload then passed in 94.2, 84.6 and 79.7 s. The serving script's default
chunk is now 8192, overridable with `FREETOKEN_PREFILL_CHUNK`.

### Live-server depth checks

Two distinct 100k prompts from the manifest were sent to the deployed server
(disk prefix cache on, 15 and 19 minutes after restart). Both passed. Cold TTFT
was 138.60 s and 165.92 s; the cached repeat of the first reused 99,904 tokens
with a 2.96 s TTFT. The journal shows the difference from the 89.6 s arm is in
the chunks themselves: 8192-token chunks took 11 to 17 s on the live server
against 6.9 to 8.9 s in the arm, with no additional delay before the first
chunk (the "Prefill batch" line is written at chunk completion, so its
throughput figure for a request's first chunk is not a chunk timing).

A second pass of the continuation workload on the live server, with the disk
prefix cache holding the first pass, completed in 51.4, 46.2 and 46.5 s per
session with identical outputs, reading 2,048 to 3,776 cached tokens per call.
Decode is therefore unchanged; the slower cold sessions are uncached prefill.
The remaining suspect for the slower live chunks is the prefix cache itself,
whose 1.4 GiB per-100k-request writes go through the same page cache that
holds the file-backed expert banks. That is measured next as `workspacecurve4-*`
(8192, finalist tree, prefix cache on with a fresh directory, same manifest).

## Prefix cache staging fix and host budget arms, 2026-09-27

The live-server gap between cache-off arms (100k cold 89.6 s) and the deployed
server (139 to 196 s) came from pinned host memory that the disk prefix cache
retained after each entry write. The cause, the fix and the live traces are in
`docs/prefix-cache-page-pressure-20260926.md`. With pageable staging (`f28f44b`)
the cache-on protocol (`stagefix1-*`) passed all 27 rows with exact parity and
ran 100k cold in 93.4, 90.0 and 88.3 s with 13.0, 10.0 and 7.9 GiB read from
NVMe, matching the cache-off arm.

### Host budget arms

On the fixed tree (`368d9c4`, same tree as `8422299`) the `pagercache1-*`
sweep tested whether a smaller host reserve or a larger pager share would leave
more page cache to the file-backed DISK banks. Three arms ran the same
protocol back to back, cache on, fresh directory each. The two non-baseline arms
held the pin budget at 27.07 GiB with `FREETOKEN_PIN_BUDGET_GB` so placement
could not change; every startup log showed 20 GPU prefill layers, 28 DISK
layers and 82 protected HOT experts.

| Arm | Budget table | 100k cold TTFT | NVMe per 100k cold | 32k cold TTFT | 32k repeat wall | 32k post-prefill decode wall | 100k post-prefill decode wall | Swap-out during run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | reserve 9.29, pin 27.07, pager 21.24 (derived) | 91.99 s | 11.8 GiB | 37.45 s | 3.98 s | 19.51 s | 25.23 s | 2.83 GiB |
| `--host-cache-reserve-gib 6` | reserve 6.00, pin 27.07, pager 24.60 (derived) | 90.95 s | 9.8 GiB | 42.49 s | 4.45 s | 24.54 s | 24.15 s | 2.73 GiB |
| reserve 4.5, `--moe-pager-budget-gib 25.3` | reserve 4.50, pin 27.07, pager 25.30 | 91.69 s | 9.7 GiB | 44.10 s | 5.93 s | 24.80 s | 25.38 s | 2.72 GiB |

All 81 responses passed with exact parity against `workspacecurve1-0`. No arm
won. The 100k cold differences are inside the run-to-run spread of the
baseline itself (89.3 to 94.6 s), total NVMe reads over each arm were 118, 124
and 126 GiB, and both non-baseline arms were slower on the 32k rows. The
defaults in `engine/host_memory.py` are unchanged.

The null result follows from what the two knobs control on this profile:

- The reserve is an accounting input, not memory the process holds. It lowers
  the governor's ceiling, and with a derived pin budget the pin share (28/50 of
  the remainder) grows with it. At reserve 6 the derived pin budget would be
  about 28.9 GiB, enough for a 21st GPU layer, which is why these arms fixed
  the pin budget.
- The served profile uses `moe_disk_pager='madvise'`, not `uffd`, so the DISK
  banks are an ordinary file mapping and the kernel page cache decides residency. The pager budget
  then only sets the CPU prefill populate ceiling (half the budget per layer:
  10.64, 12.30 and 12.65 GiB), and every DISK layer already allows all 512
  experts at the baseline value.

Page cache for the DISK banks is what physical memory leaves after the pinned
banks (26.44 GiB for 20 GPU layers plus 0.56 GiB HOT staging), the engine's
anonymous memory and the rest of the host. Giving the banks more of it means
pinning fewer layers, which trades GPU prefill layers for DISK layers and is a
placement experiment, not a cache setting.

Artifacts: `stagefix1-*`, `stagefix2-*` (reverted transient pinning) and
`pagercache1-*`, with `node4-cachepressure-summary.py` for per-request TTFT,
NVMe read, stall and parity, and `pagercache1-summary.txt`.

## HOT adaptation across idle gaps, 2026-09-27

With the prefix-cache leak fixed, the live server still ran cold 100k prefills
in 106 to 115 s against about 92 s in the qualification arms. The steady
8192-token chunks took about 7 s on both; the difference was the first two or
three chunks after an idle gap (about 19, 15 and 9 s live against 9 to 10 s).
Under `--moe-hot-adapt-aim phase` the idle ticks aim at the decode history
alone. Every request starts with a prefill, so after each gap the next prefill
began at a 26 to 27% decayed hot rate with a 1,148-swap plan and read 24 to 29
GiB from NVMe instead of about 10. The arms hid this: their 8k and 32k rows
triggered the adaptation's bandwidth back-off early, which spaced later ticks
out, and no row followed a long idle.

Two protocols reproduce live traffic, both on `a120203` with the prefix cache on
and a fresh directory per arm, and all rows passed with exact parity:

- `livelike*`: a freshly started server, the `node4-finalist-verify.py` warm-up,
  then the three 100k manifest cases each after 90 s of idle
  (`node4-livelike-driver.py`). Its baseline reproduced the live numbers.
- `idlegap*`: the fixed continuation workload, then the 27 depth rows with 60 s
  of idle before every cold request (`prefill-depth.py measure
  --idle-before-cold 60`).

Live-like 100k cold TTFT (runs 2 and 3; run 1 is page-cache warm-up after the
restart in every arm) and NVMe read:

| Sweep, arm order | Baseline | Idle ticks off | Idle off + post-prefill tick | Idle aimed at prefill |
| --- | --- | --- | --- | --- |
| livelike1 (baseline first) | 109.8 / 106.5 s, 27 / 24 GiB | 92.3 / 92.4 s, 10 / 11 GiB | | 92.6 / 92.0 s, 11 / 11 GiB |
| livelike2 (idle off first) | 109.6 / 108.4 s, 28 / 29 GiB | 93.2 / 91.4 s, 12 / 10 GiB | | |
| livelike3 (post-prefill first) | | 95.8 / 93.5 s, 12 / 11 GiB | 95.7 / 93.5 s, 14 / 12 GiB | |

Idle-gap depth protocol, means of three runs:

| Arm (sweep) | 8k cold TTFT / wall | 32k cold TTFT | 100k cold TTFT | 100k repeat wall | 100k post-prefill decode wall | Continuation walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (idlegap1) | 13.37 / 30.02 s | 37.86 s | 93.23 s | 3.88 s | 21.10 s | 99.5 / 72.8 / 79.3 s |
| baseline (idlegap2) | 13.60 / 30.78 s | 39.59 s | 98.75 s | 3.99 s | 19.90 s | 104.2 / 77.6 / 78.9 s |
| idle aimed at prefill (idlegap1) | 13.09 / 24.77 s | 36.78 s | 92.14 s | 7.44 s | 24.75 s | 93.2 / 73.1 / 73.4 s |
| idle off (idlegap1) | 11.66 / 16.97 s | 33.45 s | 93.19 s | 4.91 s | 23.16 s | 96.1 / 72.0 / 69.9 s |
| idle off (idlegap2) | 11.75 / 16.73 s | 34.22 s | 90.33 s | 5.40 s | 25.39 s | 96.9 / 71.8 / 75.4 s |
| idle off (idlegap3) | 11.52 / 18.05 s | 33.36 s | 91.20 s | 5.59 s | 24.36 s | 102.1 / 72.0 / 76.0 s |
| idle off + post-prefill tick (idlegap3) | 12.36 / 18.55 s | 30.28 s | 89.72 s | 4.22 s | 18.47 s | 100.0 / 74.1 / 66.9 s |

Aiming idle ticks at the prefill blend (an opt-in flag, since reverted) fixed
the prefill but slowed decode after cached restores. Turning idle ticks off
improved every cold row and the continuation sessions, and in all three sweeps
slowed decode right after a 100k prefill, because nothing re-aimed the set at
decode before the next tokens. One bounded decode tick at the first decode
boundary after each prefill (`--moe-hot-adapt-post-prefill-tick on`) recovers
that. The 4090 serving script now passes `--moe-hot-adapt-idle-ms 0
--moe-hot-adapt-post-prefill-tick on` (`9f9b5d2`); the engine defaults are
unchanged. The idle ticks were added for short chat turns right after startup;
on this profile the HOT set is seeded from the layer profile and reports its
fill complete at the first decode tick, so they only ran after the fill.

Deployed with `node4-finalist-deploy.sh 8192` on `9f9b5d2`; the startup log
shows `idle=off`, the post-prefill tick on, 20 GPU prefill layers and 82 HOT
experts, and `node4-finalist-verify.py` passed. The same live sequence as
before (verify warm-up, then three fresh 100k prompts 90 s apart,
`livetrace-idletick-20260927.jsonl`):

| Live 100k cold request | Before (prefix fix only) | After (idle off + post-prefill tick) |
| --- | ---: | ---: |
| first after restart | 179.6 s, 96.1 GiB | 154.0 s, 69.8 GiB |
| second | 114.5 s, 33.9 GiB | 97.3 s, 14.7 GiB |
| third | 105.9 s, 25.5 GiB | 93.2 s, 12.7 GiB |

The first request after a restart still pays the page-cache warm-up of the
file-backed DISK banks. Later cold 100k requests on the live server now run
within a few seconds of the qualification arms.

## Layer-major prefill and DISK staging, 2026-09-28

Chunked prefill runs all 48 layers for each chunk, so every layer's experts
cross PCIe once per chunk: at 100k with 8192-token chunks, 13 passes over the
26.4 GiB of pinned banks and 13 routed-union stagings of the 37 GiB of DISK
banks. Four changes, each measured on its own, cut cold 100k TTFT from about
92 s to 53 s on the 4090 with exact parity on every row.

### What changed

1. **Layer-major groups** (`--prefill-layer-major-tokens N`). Consecutive
   chunks of one request run layer by layer: every chunk of a group passes
   layer L before any reaches layer L+1, so a layer's experts move once per
   group. The scheduler prepares each chunk exactly as chunk-major serving
   would (pages, QSA and GDN metadata, snapshot tracking) from the lengths the
   previous chunk's launch leaves behind. A group stops at N tokens, at the
   prompt's final chunk, and at a chunk persisting an intermediate harness root
   (its ping-pong snapshot slot is rewritten two chunks later). Only a lone
   request with no running decode is grouped; the group drains chunk by chunk;
   an OOM aborts the request cleanly. The model runs the same operations and
   shapes per chunk as the chunk-major forward; PLE runs per chunk at its layer.
   The group's hyper-connection residual (20 KiB per token here) stays on the
   GPU, so N follows free GPU memory: 4 x 8192-token chunks ran out of memory on
   the 4090 and 32k groups fit with 4096-token chunks
   (`--prefill-layer-major-chunk 4096`, applied only to prompts that need more
   than one `--max-extend-length` chunk). On a 96 GB GPU a whole prompt fits in
   one group.
2. **Parallel DISK staging reads** (`FREETOKEN_DISK_STAGING_WORKERS`). The
   staging ring read one 32 MiB piece at a time on one thread (~1.9 GB/s in the
   trace). Reads are now dealt to N workers with two pinned slots each (pinned
   total ~64 MiB per reader); a single reader gets ~10.6 GB/s from page cache and
   ~1.4 GB/s from NVMe on node-4, eight get several GB/s on cold data.
3. **Predicted DISK staging** inside a group: every layer alternates between the
   two prefill double buffers, a background reader stages the rows each DISK
   layer routed in its previous group while the previous layer computes, and
   only the rows the prediction missed (~550 of ~14,000 per group) are staged
   on demand. Chunk c+1's attention and routing are enqueued before chunk c's
   experts, and the routed rows are read back through a fixed-size mask fenced
   by an event, so the host never waits on the whole stream.
4. **Cached DISK reads** (`--moe-disk-prefill-io cached`, now parallel): rows
   already in the page cache are read through it and cold rows with O_DIRECT.
   The DISK banks (37 GiB) exceed the page cache left for them (~31 GiB).
   Chunk-major touches popular experts every chunk, so LRU keeps them; a group
   touches each routed row about once, so buffered group reads became a 36 GiB
   sweep that evicted the next group's rows and decode's hot rows alike (32k
   cold reads 16.7-21.4 GiB against 3-13 GiB, decode after prefill 2-5 s
   slower). Direct reads leave the page cache to the rows decode uses.

Tried and reverted on the way: streaming whole DISK layers ahead (routing is
skewed, so it read the rarely used experts, the pages never in the page cache:
22-29 GiB from NVMe per 32k request and no gain), and aiming idle ticks at
prefill.

### Group trace

`FREETOKEN_LAYER_MAJOR_TRACE=1` logs per-layer GPU time and staging counters
for each group. One 32k group (8 chunks of 4096 tokens):

| Build | Group GPU time | DISK layers | Pinned layers | DISK staging on demand |
| --- | ---: | ---: | ---: | ---: |
| per-row staging, one reader | 22-27 s | 16-22 s | 5.3-5.5 s | 19.4 s (first group) |
| + predicted staging | 26.7-28.0 s | 21-22.5 s | 5.3-5.5 s | 0.8-1.4 s, 19-20.5 s background reads |
| + 8 readers | 15.7 s | 10.3 s | 5.4 s | 0.35 s, 3.9-4.2 s background reads (hidden) |

### Idle-gap protocol, final configuration

`idlegap9` (final configuration first, then buffered reads, then baseline;
continuation workload plus 27 depth rows with 60 s idle before each cold
request). All 81 depth rows and 9 continuation sessions passed with exact
parity.

| Arm | 8k cold TTFT / wall | 32k cold TTFT / wall | 100k cold TTFT / wall | 32k repeat wall | 100k post-prefill decode wall | Continuation walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final (cached reads) | 5.97 / 10.81 s | 16.94 / 21.77 s | 53.16 / 60.36 s | 3.92 s | 17.52 s | 100.9 / 73.2 / 65.0 s |
| buffered reads | 6.73 / 13.23 s | 17.69 / 24.62 s | 54.80 / 62.47 s | 4.22 s | 20.11 s | 93.2 / 79.1 / 78.8 s |
| baseline | 12.11 / 17.23 s | 33.77 / 37.48 s | 93.69 / 98.31 s | 4.19 s | 18.04 s | 94.5 / 72.8 / 69.0 s |

Earlier steps on the same protocol: layer-major alone (32k groups) 100k cold
68.0-69.8 s against 89.3-91.1 s baseline in both arm orders (`idlegap4`,
`idlegap5`, `idlegap7`); 8 readers alone, chunk-major, 69.7 s (`idlegap8`).

### Scaling

The pieces are sized by the machine rather than the model: the group budget by
free GPU memory (the residual is `hc_count x hidden x 2` bytes per token), the
staging readers by storage and page-cache bandwidth, and the double buffers
already exist. With more GPU memory a group covers the whole prompt (one expert
pass per prompt) and more layers stay GPU-resident; a larger model with more of
its experts on DISK is bound by the same staging path, which now reads in
parallel, predicts a group's rows, and keeps cold rows out of the page cache.

### Deployment

Deployed with `node4-finalist-deploy.sh 8192` on `c841a64`: the startup log
shows `file_io=cached, workers=8`, 20 GPU prefill layers, 28 DISK layers and 82
HOT experts; `node4-finalist-verify.py` passed. The live sequence (verify
warm-up, then three fresh 100k prompts 90 s apart,
`livetrace-layermajor-20260928.jsonl`) ran in 57.3, 53.6 and 53.3 s TTFT, against
154.0, 97.3 and 93.2 s for the same sequence on the previous deployment. Shmem
stayed at 27.26 GiB with no swap; prefill read about 60 GiB from NVMe per 100k
request, most of it cold rows by direct I/O.

## Prefill MoE kernel, tail chunk and PLE staging, 2026-09-29

Three more changes on the layer-major build, each with exact parity on every
depth row and continuation session. Cold 100k TTFT on the idle-gap protocol
went from 45.5 s to about 24 s.

### What changed

1. **v2 prefill NVFP4 MoE kernel** (`1f2619e`, deployed). The v1 kernel loaded
   the even and odd K columns of the activations as two stride-2 gathers and
   decoded every e2m1 weight through a lookup-table load. v2 loads the
   activations contiguously, splits the two nibble planes in registers, and
   decodes e2m1 arithmetically (the magnitude bits form an fp16 pattern worth
   value x 2^-14). The values and the order of both dots are unchanged, so the
   output is bit-identical to v1. At the Qwen3.8-Flash expert shapes (512
   experts, top-10) a 4096-token chunk takes 5.45 ms with 64x128 tiles, 4 warps
   and 2 stages, against 16.1 ms for v1 at its best tiles. Batching the expert
   GEMM across chunks (`FREETOKEN_LAYER_MAJOR_MOE_TOKENS`, `6631a9d`, off by
   default) left the expert GEMM time unchanged with v2 (`lmkernel1`).
2. **The final chunk joins the last group** (`d1b6e7a`). A 100k prompt split
   into three 32,768-token groups and a 1,655-token tail. The tail ran
   chunk-major and streamed every DISK layer's experts again: about 4-5 s for
   2% of the tokens. The group budget may now be overrun by the prompt's final
   chunk (at most one chunk); that group costs 0.5-0.8 s more.
3. **PLE rows staged on a host thread** (`6839780`, then `06079df`). The uring
   PLE backend hashed each chunk's n-grams on the GPU and read the row ids back
   (`row_ids.cpu()`), which drained every queued kernel of the group once per
   chunk while the host deduplicated and read the rows (16 row ids per token,
   two small O_DIRECT reads per unique row). Each chunk is now hashed from the
   host prompt (the tested host hash) and its rows read into a ring of slots in
   the staging bank, above the rows ordinary fills use. The first version
   staged one group at a time and only gained layers 0-1 of head start; the
   request-level version plans every remaining chunk of the prompt, so the
   next group's rows are read while the current group runs.

### Group profile

One 32k group with v2 (`lmple1`, torch profiler): 8.5 s wall, GPU kernels busy
6.0 s. The prefill MoE kernel took 2.0 s (about 78 TFLOPS on the rows it
computes), host-to-device copies 2.6 s (partly overlapped), dense bf16 GEMMs
1.5 s, attention 0.4 s and GDN 0.3 s. PLE spans per group:

| Build | PLE span, first group | PLE span, later groups | Group GPU time, later groups |
| --- | ---: | ---: | ---: |
| v2 (`lmple1`) | 1.8-2.3 s | 1.8-2.3 s | 9.0-9.6 s |
| + tail join, group stager (`lmtrace7`) | 1.5-1.8 s | 1.5-1.8 s | 8.5-9.9 s |
| + request stager (`lmtrace10`) | 1.5-1.8 s | 0.14-0.24 s | 7.1-7.9 s |

The first group of a request still waits for its PLE rows: the reads are bound
by IOPS, and the thread starts with the request. `fb1b1b6` reads the data and
scale stores concurrently for large fills (`lmtrace11`, pending).

### Idle-gap protocol

Each pair runs in both orders. Cold 100k TTFT, mean of 3:

| Run | First arm | Second arm |
| --- | --- | --- |
| `idlegap10` | v2 29.44 s | v1 45.60 s |
| `idlegap11` | v1 45.50 s | v2 30.17 s |
| `lmstage2` | tail + group stager 25.90 s | v2 29.64 s |
| `lmstage3` | v2 29.54 s | tail + group stager 25.98 s |
| `lmstage4` | tail + group stager 26.68 s | + request stager 23.93 s |

The v1 arms run the v2 build with `FREETOKEN_NVFP4_PREFILL_KERNEL=v1` (v2's
tiles). 32k cold TTFT: v1 14.4 s, v2 9.4-9.5 s, later builds 9.0-9.2 s. On the
live-like driver (verify warm-up, then three fresh 100k prompts 90 s apart) v2
ran 30.4 s against 45.8 s for v1 (`livelike9`).

Continuation walls follow arm order: the first arm pays 105-120 s for session 1
and the second 91-97 s, whatever the build. The tail and group-stager build
also paid about 107-109 s as the second arm in `lmstage3`; its code does not
run for the continuation prompts (about 2k tokens, one chunk), and `contab1`
(alternating fresh servers, two passes each) is pending to settle it.

### Tried: SwiGLU in the gate/up GEMM epilogue

`562458a` (branch `dev/moe-swiglu-fusion`, not on this branch) computes the gate and up halves of the same columns in one program,
rounds both to bf16 as the unfused kernel stores them, and applies flashinfer's
act_and_mul instruction sequence (sm_89 fast-math SASS: FMUL.FTZ, MUFU.EX2,
FADD.FTZ, MUFU.RCP, FMUL.FTZ) in inline PTX. It is bit-identical (all 65,536
bf16 gate values against flashinfer, and whole MoE layers) but 1.5-2.3% slower
than the separate pass (6.15 against 6.05 ms per 4096-token chunk), so it was
left out. The expert GEMMs are
compute-bound; a fused MoE megakernel could remove at most the activation,
align and top-k sum kernels (about 0.15-0.25 s per 32k group), and the top-k
sum cannot be fused without changing its summation order.

### Deployment

`1f2619e` (v2) is on the serving branch. The tail and PLE staging changes
deploy after `contab1` and `lmstage5`.

A node-4 hard hang at 15:40 UTC on 2026-09-28 (no panic record, journals
closed uncleanly, idle between benchmark requests) interrupted the first run
of these benchmarks; they were rerun after the reboot.

## Group size, decode breakdown and kernel probes, 2026-09-29

### Confirmations

- Request-level PLE stager, reversed order (`lmstage5`): 100k cold 24.14 s as
  the first arm against 26.45 s for the group stager (`lmstage4`: 23.93 s
  against 26.68 s).
- Concurrent PLE data and scale reads (`lmtrace11`): the first group's PLE span
  went from 1.5-1.8 s to 0.79-0.97 s; later groups 0.08-0.16 s.
- Continuation walls (`contab1`, four fresh servers alternating v2 and the
  tail/stager build, two passes each, 24 sessions with exact parity): fresh
  server totals 253.6 and 247.0 s (v2) against 292.6 and 244.3 s; warm pass
  totals 222.4 and 156.3 s against 158.0 and 152.9 s. The earlier gap was
  cold-start variance.
- One `lmstage5` depth row (8k repeat) diverged while CPU-heavy tests ran on the
  host. Cold experts run on the CPU (W4A8), with different rounding than the GPU
  path, and HOT adaptation timing decides which experts run where, so host load
  can change a greedy token. The same build matched in three other runs.
  Benchmarks need a quiet host.

### Group size

`lmgroup1`-`lmgroup3`, live-like driver (verify warm-up, three fresh 100k
prompts 90 s apart) on `main`:

| Arm (order) | 100k cold TTFT | Warm decode | NVMe per 100k |
| --- | ---: | ---: | ---: |
| 32k groups, `--memory-ratio 0.87` (1st) | 23.04 s | 17.2 tok/s | 50-65 GiB |
| 64k groups, `--memory-ratio 0.87` (2nd) | 21.86 s | 16.5 tok/s | 37-39 GiB |
| 32k groups, 0.90 (3rd) | 22.71 s | 17.4 tok/s | |

64k groups and whole-prompt groups ran out of GPU memory at the default 0.90
(the automatic sizing gives every byte above the `1 - memory_ratio` headroom to
KV and expert slots); lowering the HOT budget did not free any. At 0.87 the
64k groups fit, every row matched, and the prompt makes one expert pass fewer.

### Decode breakdown

`--moe-step-timing` on the continuation workload (`lmdecode1`): a decode step
takes about 30 ms, of which the 19 GPU-resident layers take 3.0 ms and the 29
DISK layers 26.1 ms (about 0.9 ms each, against 0.16 ms). CPU expert compute is
0.29 ms per DISK layer (about 1.7 cold experts, 137-313 MB of weights per
step); host-device copies are 0.56 ms per step. Decode is bound by the GPU
waiting on the CPU partial of each DISK layer. Fetching predicted cold experts
to the GPU a layer ahead would remove most of that wait, but it changes which
experts run on the GPU (W4A16) and the CPU (W4A8), so it needs a new parity
reference.

### Predicting cold experts

`FREETOKEN_ROUTE_DUMP` (eager decode, `--cuda-graph-max-bs 0`) recorded each
layer's router input and logits for about 3,800 decode steps of the
continuation workload (`routedump1`); `scripts/routedump-analyze.py` measures
how many of a DISK layer's cold routes (experts not HOT) a fetch of F predicted
experts would cover. There are 3.2 cold routes per DISK layer per token.

| Predictor | F=1 | F=2 | F=4 | F=8 | F=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| next layer's router on the previous layer's router input | 9.8% | 17.3% | 28.5% | 42.8% | 58.9% |
| previous token's cold experts at the layer | 3.1% | 5.2% | 7.6% | 8.9% | 9.0% |

Fetching predicted cold experts to the GPU does not pay here: half the cold
routes need about 10 experts (about 28 MB per layer over PCIe, slower than the
layer's 0.9 ms today), and a layer still waits on the CPU while any cold route is
uncovered. The lever that remains is the CPU round trip itself: a DISK layer
takes 0.9 ms against 0.16 ms for a GPU-resident layer, of which the CPU expert
compute is 0.29 ms.

### Kernel probes

- Tile sweep of the v2 prefill MoE kernel (`moebench/bench_tiles2.py`, 105
  configurations at M = 4096, 8192, 16384, all bit-identical): the shipped tiles
  are within 0-3% of the best at every M. The kernel is faster per token at
  larger M (4.44 against 5.65 ms per 4096 tokens at M = 16384), which only
  cross-chunk expert batching reaches, and that does not fit on the 4090.
- FP8 (e4m3) tensor-core MMA with fp32 accumulation (`moebench/bench_fp8.py`,
  weights cast to e4m3 after dequantization, activations quantized per row per
  K tile): 5.05 against 5.65 ms per 4096 tokens at M = 4096 and 4.06 against
  4.45 ms at M = 16384, with 5.5% relative RMS error on the MoE layer output.
  The kernel is bound by dequantization and data movement, not MMA rate; not
  pursued.

### Deployment

`main` (`0b67820`) serves from `/disks/nvme-02/src/freetoken-serve`; the
live-like driver measured 100k cold 23.47 s on it (`lmgroup1`), against 30.4 s
for v2 (`livelike9`).
