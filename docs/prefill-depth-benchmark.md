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


## Decode fault-guard boundary diagnosis

A follow-up on merged runtime `11654ed` enabled `--moe-step-timing`,
`--moe-collect-stats` and ten-step decode logs for the 8192/fixed-1000 and
2048/automatic arms. All six responses passed and matched exactly across arms,
including request hashes, complete text, reasoning, finish reasons and usage.
Artifacts use `diag1-*` in the same private directory.

The 8192 cold-answer windows reported one prefetch guard trip and no skipped
expert advice, although each window's reported major-fault rate was below the
configured 2000 faults per decode step ceiling. Code inspection and two behavior
tests confirmed that the guard's initial baseline included
startup faults, and its next sample after prefill counted all intervening prefill
faults as one decode interval. These counters are process-wide, so they cannot
attribute faults exclusively to CPU expert pages.

The experimental policy establishes a baseline at the first decode and invalidates that baseline
at existing prefill/cache-reset boundaries. It preserves measured decode history,
recent expert touches and any active 256-step pressure hold. Genuine excessive
faults between consecutive decode steps still activate the guard. Both new behavior tests
fail before the change; 37 targeted timing, prefetch, lookahead and statistics
tests pass on node-4 Linux after it. These tests establish the changed accounting,
not a performance improvement. The original rationale in `bench/RESULTS.md`
explicitly used prefill faults to detect loss of the resident working set.
Excluding those faults therefore changes that pressure policy and needs measured
qualification before deployment.

The instrumented 8192 cold-answer rate was 14.97 tok/s, versus 9.15 tok/s in the
previous uninstrumented experiment; the instrumented 2048 rate was 8.57 tok/s.
Synchronization and counters alter execution, and these samples also vary across
fresh starts. They do not establish a speedup or qualify a new serving profile.
An uninstrumented comparison of the fix and unchanged controls is required.


## First uninstrumented fault-policy comparison

The `guard1` comparison ran the experimental policy (`39f5dc2`) at 8192 chunks
and fixed interval 1000, unchanged `11654ed` with the same chunk/interval, then
unchanged `11654ed` at the selected 2048/automatic settings. Other settings and
the 100k manifest were held fixed. Each arm restarted from empty prefix state.

| Policy / chunks / interval | Cold TTFT | Cold wall | Cold answer decode | Repeat TTFT | Repeat wall | Repeat decode | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode-only faults / 8192 / 1000 | 171.37 s | 178.77 s | 11.10 tok/s | 5.86 s | 8.47 s | 31.56 tok/s | 26.61 tok/s |
| Original / 8192 / 1000 | 160.80 s | 167.78 s | 11.79 tok/s | 2.65 s | 6.04 s | 24.64 tok/s | 25.51 tok/s |
| Original / 2048 / auto | 268.25 s | 278.53 s | 8.17 tok/s | 1.70 s | 5.68 s | 20.77 tok/s | 24.82 tok/s |

All nine responses passed and matched exactly across arms in request hashes,
complete text, reasoning, finish reasons and usage. Cold hits were zero and
repeats reused 99,904 tokens. The controller completed and restored the selected
serving service. Artifacts use `guard1-*` in the same private results directory.

The candidate reduced cold first-text latency by 36% relative to the selected
profile and had higher client-observed decode rates in all three phases in this
sample. However, the cached repeat's total wall time increased by 49%, due to
its longer first-text wait. Against the same-size unchanged arm, repeat decode
improved by 28% while total repeat wall increased by 40%. These observations
separate generated-token rate from whole-request performance; they do not
establish a uniformly faster profile. The ten-second cold-TTFT difference
between same-size arms also cautions against attributing a single fresh-start
measurement to the decode-only policy. A follow-up tests the candidate at 2048
and repeats the 8192 arm before continuation qualification or deployment.


## Fault policy at the selected chunk size

The next completed arm kept 2048 chunks and the automatic adaptation interval,
changing only the fault-accounting policy. Its three responses passed and
matched the preceding unchanged 2048 arm exactly, including full messages,
request hashes, finish reasons and usage.

| Policy | Cold TTFT | Cold wall | Cold answer decode | Repeat TTFT | Repeat wall | Repeat decode | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, preceding control | 268.25 s | 278.53 s | 8.17 tok/s | 1.70 s | 5.68 s | 20.77 tok/s | 24.82 tok/s |
| Decode-only faults | 267.22 s | 274.44 s | 11.33 tok/s | 2.71 s | 5.40 s | 30.74 tok/s | 27.26 tok/s |

Cold first-text latency was essentially unchanged, as expected for a policy
applied at decode boundaries. All three decode-rate estimates improved in this
sample. The repeat's first-text wait increased, but faster generation reduced
its total wall time by about 5%. This is one fresh-start observation, not an
isolated causal estimate or completed qualification.

The arm's artifacts are `guard2-0-chunk-2048-cap-0.0-interval-auto.*`. The
controller then failed before starting its second arm because systemd still
had the shared transient service name loaded. Recovery restored serving; the
completed measurements were retained. Only the unrun 8192 repeat was launched
under `guard2b`, using a distinct service name. Continuation qualification also
uses a distinct service per arm to avoid this launch conflict.


The repeated 8192/fixed-1000 candidate (`guard2b`) subsequently completed and
passed all three requests. It matched both completed 2048 arms exactly in full
responses, requests, finish reasons and usage. With the same experimental fault
policy in both chunk sizes:

| Chunks / interval | Cold TTFT | Cold wall | Cold answer decode | Repeat TTFT | Repeat wall | Repeat decode | Independent wall | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 / auto | 267.22 s | 274.44 s | 11.33 tok/s | 2.71 s | 5.40 s | 30.74 tok/s | 21.33 s | 27.26 tok/s |
| 8192 / 1000 | 147.38 s | 153.59 s | 13.19 tok/s | 2.88 s | 5.42 s | 32.67 tok/s | 21.26 s | 27.32 tok/s |

The larger chunk reduced cold first-text latency by 45%, with repeat wall within
0.4% and essentially unchanged independent decode in this pair. The earlier
8.47-second candidate repeat is retained above; fresh-start variability remains
part of the evidence. The controller restored healthy serving, and the matched
continuation qualification began only after all six follow-up responses passed
exact parity. This is still a candidate profile pending that qualification.


## Continuation qualification of the fixed-interval candidate

`guardqual2` compared unchanged `11654ed` at 2048/automatic against `39f5dc2`
at 8192/fixed-1000. Each ran the existing three-conversation, three-turn protocol
with its original prompts, budgets and graders. Conversation 1 was the prescribed
warm-up. Each arm had its own initially empty 2 GiB disk-prefix cache and lazy
restore enabled. All 18 responses passed and matched in complete requests,
messages, finish reasons, prompt counts and output counts.

| Profile | Warm-up | Measured conversation 2 | Measured conversation 3 | Measured mean |
| --- | ---: | ---: | ---: | ---: |
| Original / 2048 / auto | 95.54 s | 72.47 s | 66.52 s | 69.50 s |
| Decode-only faults / 8192 / 1000 | 109.04 s | 72.54 s | 79.56 s | 76.05 s |

The candidate's measured mean was 9.4% slower. This prevents selecting the
fixed-interval profile despite its favorable long-prefill sample. Saved startup
adaptation logs show another material policy difference: the first automatic
prefill boundary consumed 12 ticks and planned 1,148 swaps, while fixed cadence
consumed two ticks and planned 386. The initial prompts were about 2.1k tokens.
This makes reduced initial HOT-cache adaptation a plausible contributor, not a
proven sole cause. The larger-chunk candidate needs a comparison retaining the
selected automatic cadence before introducing further runtime changes.

The queued coding test was stopped before taking the GPU, preserving its frozen
protocol for a qualified finalist. A follow-up runs the original continuation
workload followed by the 100k depth workload in each fresh server, candidate
first and control second, with automatic cadence and disk-prefix persistence
disabled in both. This holds cadence constant and measures long prefill after
ordinary conversation activity, rather than relying on one clock phase from an
otherwise fresh server. No new serving default has been selected.


## Automatic cadence after ordinary continuation

`auto1` completed the continuation protocol followed by the 100k workload on
fresh servers: `39f5dc2`/8192 first, `11654ed`/2048 second, automatic adaptation
and disk-prefix persistence disabled in both. All 18 continuation responses and
all six depth responses passed, with exact full-request, message, finish and
usage parity. The prescribed first conversation was excluded from the measured
continuation mean. Cold refers to the prefix cache, not the OS page cache.

| Metric | Original / 2048 | Decode-only faults / 8192 |
| --- | ---: | ---: |
| Measured continuation mean | 65.99 s | 67.70 s |
| 100k cold TTFT | 252.41 s | 125.91 s |
| 100k cold wall | 255.83 s | 129.63 s |
| Cold answer decode | 24.25 tok/s | 22.17 tok/s |
| Repeat TTFT | 2.55 s | 1.54 s |
| Repeat wall | 5.53 s | 4.63 s |
| Repeat decode | 27.66 tok/s | 26.64 tok/s |
| Independent request wall | 19.05 s | 20.92 s |
| Independent request decode | 31.23 tok/s | 27.24 tok/s |

The candidate halved cold first-text latency, but continuation averaged 2.6%
slower and independent decode was 12.8% slower. It is not qualified for
promotion. Both repeats reused 99,904 tokens; both cold requests had zero hits.
A two-second host observer covered at least 95% of both cold windows. On-node
comparisons found higher memory pressure, I/O pressure and disk reads for the
candidate. Detailed host counters remain on node-4. These are host-wide
observations, with phase endpoints observed up to two seconds late, not proof
that one subsystem caused the slowdown.

Source inspection shows that increasing `host_cache_reserve_gib` reduces both
derived pinned-expert and pager budgets (28:22 split after fixed costs). It can
therefore increase disk residency rather than simply adding free page cache.
The next isolated screening comparison keeps `11654ed` and automatic cadence
in both arms and changes only chunks from 2048 to 4096. No larger default or
fault-policy change has been selected.


## Intermediate chunks with the original fault policy

`mid1` isolated chunk size on unchanged `11654ed`: 2048 first, then 4096,
automatic cadence and no disk-prefix persistence in both. Each fresh server ran
the original continuation protocol before the 100k workload. All 18 continuation
responses and six depth responses passed with exact full-response, request,
finish and usage parity. The first conversation remained warm-up.

| Metric | 2048 | 4096 |
| --- | ---: | ---: |
| Measured continuation mean | 66.51 s | 67.40 s |
| 100k cold TTFT | 253.59 s | 169.61 s |
| 100k cold wall | 256.96 s | 173.51 s |
| Cold answer decode | 24.71 tok/s | 21.19 tok/s |
| Repeat TTFT | 2.57 s | 2.63 s |
| Repeat wall | 5.84 s | 5.51 s |
| Repeat decode | 25.20 tok/s | 28.64 tok/s |
| Independent request wall | 19.29 s | 27.74 s |
| Independent request decode | 30.71 tok/s | 19.44 tok/s |

The 33% cold-TTFT gain does not qualify this profile: independent decode was
37% slower and cold-answer decode was 14% slower. Continuation differed by 1.3%
in this small sample. Both cold requests had zero prefix hits; both repeats
reused 99,904 tokens. The host observer covered at least 95% of both cold windows
and again found higher memory pressure, I/O pressure and disk reads for the
larger chunk. Host-wide counters remain on node-4 and are not causal attribution.

The controller completed successfully and restarted the selected 2048 serving
service. The prepared three-repetition depth sweep remains unstarted. Further
work should investigate the memory/cache-pressure tradeoff before promoting a
larger default. This comparison did not use the experimental fault policy.


## Increasing host reserve at 4096 chunks

`reserve1` tested unchanged `11654ed` with automatic adaptation and no persistent
prefix cache. The order was 4096 with 16 GiB host reserve, 4096 with default
reserve, then the selected 2048/default-reserve control. Preflight used actual
model geometry and the saved startup budget; the running candidate matched its
predicted reduction in pinned layers. No startup pressure warning appeared for
the candidate. Each fresh server ran continuation before the 100k workload.
All 27 continuation responses and nine depth responses passed with exact parity.

| Metric | 4096 / reserve 16 GiB | 4096 / default reserve | 2048 / default reserve |
| --- | ---: | ---: | ---: |
| Measured continuation mean | 65.63 s | 69.79 s | 70.55 s |
| Cold TTFT | 180.78 s | 169.37 s | 258.03 s |
| Cold wall | 184.71 s | 172.97 s | 261.68 s |
| Cold answer decode | 20.93 tok/s | 23.03 tok/s | 22.75 tok/s |
| Repeat TTFT | 1.66 s | 2.76 s | 2.62 s |
| Repeat wall | 4.57 s | 5.55 s | 5.70 s |
| Repeat decode | 28.35 tok/s | 29.62 tok/s | 26.73 tok/s |
| Independent wall | 21.65 s | 22.61 s | 19.67 s |
| Independent decode | 26.69 tok/s | 25.04 tok/s | 30.13 tok/s |

The larger reserve improved continuation and cached-repeat wall time, but
independent decode remained 11% slower than the selected control. Cold-answer
decode was also slower. It is not qualified for promotion. All cold requests
had zero prefix hits and all repeats reused 99,904 tokens.

The observer covered at least 95% of every cold window and 80% of every
independent-decode window, with endpoints observed up to two seconds late.
During cold prefill, the larger reserve had higher host memory pressure, I/O
pressure, disk reads and read rate than both controls. During independent
decode, its I/O pressure and reads were no higher than either control, but
memory pressure was higher. Detailed host counters remain on node-4. These
host-wide comparisons are observational; neither lower I/O nor a larger
configured reserve establishes preserved decode performance. The experiment
completed successfully and restarted the selected serving service.


## Decode-only fault accounting at 4096 chunks

`guardmid1` compared patched `39f5dc2` at 4096 and 2048, then unchanged
`11654ed` at 2048. All used automatic adaptation, default host reserve, zero
persistent prefix cache, and the same continuation-before-100k protocol.
Native library hashes matched. All 27 continuation and nine depth responses
passed with exact full-request, output, finish and usage parity.

| Metric | Patched / 4096 | Patched / 2048 | Original / 2048 |
| --- | ---: | ---: | ---: |
| Measured continuation mean | 67.07 s | 69.23 s | 68.49 s |
| Cold TTFT | 163.88 s | 254.74 s | 257.11 s |
| Cold wall | 167.48 s | 258.16 s | 260.99 s |
| Cold answer decode | 22.92 tok/s | 24.35 tok/s | 21.24 tok/s |
| Repeat TTFT | 1.58 s | 1.77 s | 1.59 s |
| Repeat wall | 4.50 s | 5.25 s | 4.67 s |
| Repeat decode | 28.19 tok/s | 23.95 tok/s | 26.76 tok/s |
| Independent wall | 20.95 s | 19.81 s | 19.96 s |
| Independent decode | 27.30 tok/s | 29.65 tok/s | 29.32 tok/s |

The patched 4096 profile cut cold TTFT by 36% relative to the selected control,
but independent decode remained 7% slower. The patched 2048 repeat was also
slower than the unchanged control. These single fresh-start samples do not
qualify either profile. All cold requests had zero prefix hits; repeats reused
99,904 tokens. The controller completed and restarted the selected service.

Cold pressure coverage exceeded 95% and independent-decode coverage exceeded
80% for each arm. The larger patched chunk had higher cold memory/I/O pressure
and read rate than both controls; total cold reads were higher than patched
2048 but no higher than original 2048. During independent decode, all four
pressure/read comparisons were no higher than either control, despite lower
decode throughput. These host-wide observations therefore do not establish
I/O pressure as the sole cause. Detailed counters remain on node-4.

A separate workspace investigation found that CPU-prefill scratch remains
allocated after use. Historical instrumented logs confirm CPU batch use below
the 1024-token staging threshold in both chunk profiles, including chunks
larger than 128 tokens. Deferring allocation can avoid unused startup memory,
but short continuations can allocate it again; persistent memory savings need
further lifecycle work and validation.
