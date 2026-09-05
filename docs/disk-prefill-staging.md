# Bounded file staging for GPU prefill

`--moe-disk-prefill staged` adds an opt-in path for native NVFP4 expert banks:
read exactly the router's selected expert union into two reusable pinned
buffers, then run the existing GPU prefill GEMM. CPU prefill remains the
default. Smaller chunks still use CPU execution, controlled by
`--moe-disk-prefill-min-tokens` (default 1,024).

On node-4's RTX 4090, the balanced prototype comparison reduced mean client
wall time from 17.57 to 9.45 seconds at 2,060 prompt tokens and from 34.93 to
16.86 seconds at 4,108 tokens. Every long-prompt pair improved. At 524 tokens,
staging was variable and approximately tied with CPU. The integrated serving
comparison below supports a 1,024-token crossover for prefill. The longer-output
gate found a 6.8% JSON response-time regression despite faster prefill. Keep
this path experimental until that post-prefill decode behavior is understood.

## Execution and memory contract

The transport preserves packed weights, all scale bytes, expert IDs, router
weights, activation, and the GPU GEMM. It performs no expert substitution,
route truncation, or extra quantization. GPU execution can differ numerically
from CPU execution; the source checkpoint and router policy are unchanged.

`DiskPrefillStaging` reads authoritative file ranges with `preadv` into two
32 MiB pinned buffers. The allocation is fixed at 64 MiB, independent of layer
size and request length. There is no whole-layer host mirror. A CUDA event
prevents host reuse before each buffer's DMA copy finishes. Copies and the
following GEMM use the caller's current stream. Engine shutdown synchronizes
pending staging transfers before releasing the cache.

The host governor charges the ring before fitting expert banks, alongside
CPU fallback workspaces and the CPU populate-read buffer. This also applies
when DISK placement is automatic and no HOT budget was explicitly supplied.
Sparse copies borrow GPU slots `[0, E)` without assigning cache ownership,
including on layers without HOT partitions. Uncopied experts remain misses;
protected HOT rows retain their mappings and bytes. Startup and cache resizing
validate enough unprotected space for the active prefill buffers.

At each chunk boundary, the threshold selects CPU or staged execution and
updates the pinned-layer buffer schedule. This keeps later asynchronous
pinned copies outside the staged layer's scratch buffer. The CPU fallback
retains its batched executor, input reuse, coalescing, and task-size checks.
Staging requires CUDA, native `nvfp4`, ordinary file-backed HostBanks, and CPU
DISK decode. UFFD sources, alternate quantized layouts, and custom model cache
factories are rejected for this path.

When HOT adaptation is enabled, the staged path records the same original
route observations, history selection, normalization, and prefill token clock
as CPU/HOT split prefill. Only a scratch copy of IDs is remapped for that
observation. Diagnostic route counts remain gated by their collection flags.
The required selected-union readback is part of transport, not telemetry.
When statistics are enabled, `decode_miss_stats()` reports staged transfer
bytes separately as `disk_prefill_staged_h2d_bytes`; pinned-overlap counters
retain their existing meaning.

Published HOT experts now reuse their existing GPU weights: one fused byte
copy gathers all six banks into scratch, and only the remaining expert rows
are read from disk. This adds small index metadata, with no weight workspace.
The publication protocol clears host owners before a worker can overwrite a
retired slot. Reuse consults those owners, since the ordinary slot map can
still name the retired expert. Copies run on the current stream before the
same GPU GEMM, retaining original expert IDs and arithmetic.

Unavailable fused copy descriptors or the copy-ablation flag retain the full
file-staging path. Optional `disk_prefill_staged_d2d_bytes` counts the reused
bytes, while the H2D counter reports only bytes actually staged from files.
Both counters are gated and reset together. This reuse change still needs
its own non-debug whole-response gate; the earlier records below predate it.

## Balanced CPU comparison

Two server starts used identical prompts, with each prompt's CPU/staging order
reversed in the second start. Both orders occur within each workload in each
start. Warmup requests are excluded from the table. Runtime: `8412464` plus
the recorded prototype hook, before the serving integration.

| Prompt tokens | Pairs | CPU wall time | Selected staging wall time | Change |
| ---: | ---: | ---: | ---: | ---: |
| 76 | 8 | 1.981 s | 1.930 s | CPU in both arms |
| 524 | 8 | 6.152 s | 6.229 s | 1.3% slower, variable |
| 2,060 | 8 | 17.573 s | 9.450 s | 46.2% shorter |
| 4,108 | 4 | 34.933 s | 16.864 s | 51.7% shorter |

The maximum chunk is 2,048 tokens, so the two larger workloads also exercise
CPU fallback for their final 12-token chunks. All 32 timing pairs, including
four short-prompt, 192-token decode controls, returned identical text and
usage. Decode controls execute CPU prefill in both arms. They had large order
effects; no decode throughput gain is claimed.

Both modes passed 7/8 long fidelity checks in each start. Every fidelity
prompt crossed the staging threshold. The code-trace question failed in both:
CPU returned `100`, selected staging returned `108`, and the expected answer
was `68`. The other seven answers matched. This is a small regression check,
not evidence of unchanged quality across arbitrary prompts.

The model was Qwen Flash NVFP4, with 14 CPU workers, 20 pinned expert layers
(26.44 GiB), 28 DISK layers (37.02 GiB), and 82 protected experts per DISK
layer. Diagnostic collection, GPU step timing, HOT adaptation and persistence,
and disk KV reuse were off. Both arms allocated the same staging ring. The
KV cache was naive, with no reused tokens. io_uring PLE, CPU input reuse, and
selective pinned transfers up to 128 tokens were enabled.

The [balanced record](../bench/results/4090-disk-cpu-balanced-20260905.json)
retains both starts, exact prompts, per-request results, arguments, journals,
and the complete driver. `bench/selected-disk-prefill-wall.py` provides the
streaming client and accepts explicit sizes for crossover measurements.

## Integrated serving crossover and adaptation

Two further starts at `3f0d438` exercised the integrated CLI, transport, memory
governor, and buffer schedule. A benchmark selector changed only the chunk
threshold: 512 in the staged arm, effectively infinite in the CPU arm. Both
arms reserved the same ring. Each prompt's order was reversed across starts;
each table cell contains four measured requests, excluding warmup.

| Prompt tokens | CPU wall time | Staged wall time | Change |
| ---: | ---: | ---: | ---: |
| 76 | 2.104 s | 2.141 s | CPU in both arms |
| 780 | 7.933 s | 7.043 s | Mixed: one prompt regressed in both orders |
| 1,036 | 9.444 s | 6.621 s | 29.9% shorter, all pairs improved |
| 1,548 | 13.959 s | 9.400 s | 32.7% shorter, all pairs improved |
| 4,108 | 34.032 s | 15.339 s | 54.9% shorter, all pairs improved |

These results select the default 1,024-token threshold. The 780-token mean
improved, but the per-prompt regressions make it a poor default. All 24 timing
pairs matched text and usage. The eight long fidelity probes again scored
7/8 per mode per start, with the same `100` versus `108` code-trace failure.

The short-prompt decode controls use CPU in both arms, yet their pooled wall
mean was 35% worse in the arm labelled selected. First-occurrence and
between-start effects remain substantial. These controls cannot establish
decode performance after staged prefill. The long-output client warms complete
responses in both modes and checks whole-response wall time separately.

The [integrated record](../bench/results/4090-disk-staged-crossover-20260905.json)
retains every request, including those unfavorable controls. The final runtime
changes at `99bfb4b` select the measured threshold, validate source ownership
earlier, and separate the gated DISK transfer counter from pinned counters.

A real CLI run at `99bfb4b` then used automatic HOT adaptation, phase histories,
and the 1,024-token threshold without an execution hook. Diagnostics confirmed
prefill history updates, token-clock advancement, and adaptation ticks and
swaps. The eight long questions scored 7/8, and all three short post-staging
questions passed. The original serving checkout was restored and returned a
real `OK` completion. This run enables diagnostics for correctness evidence;
its wall times are not performance measurements. Its complete
[record](../bench/results/4090-disk-staged-adaptation-20260905.json) is retained.

## Longer completions: a throughput regression remains

Two starts at `30f8249` generated complete warmup responses in both modes,
then reversed every measured prompt's order across starts. Serving code is
identical to `99bfb4b`. The selector used a 1,024-token threshold, with
diagnostics and adaptation off, fixed HOT placement, and no KV reuse.

| Workload | Prompt tokens | Output tokens | CPU wall | Staged wall | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy 32 records into JSON | 1,844 | 383 | 38.060 s | 40.632 s | 6.8% slower |
| Explain the source excerpt | 1,880 or 1,881 | 192 | 24.822 s | 18.588 s | 25.1% shorter |

Each cell contains four measured responses. All twelve JSON responses,
including warmups, preserve every value and the requested key order. All
measured usage counts match within pairs. The four prose pairs differ in
text, and this probe does not assign a broad prose-quality score.

JSON first-token time improved from 17.283 to 12.108 seconds, but subsequent
decode grew from 20.776 to 28.524 seconds. Its two first-start pairs improved;
both reversed-start pairs regressed. Three of four prose pairs improved.
This is why a prefill-only gain does not qualify overall throughput.

The [long-output record](../bench/results/4090-disk-staged-long-output-20260905.json)
retains all outputs and timings. The original clean serving checkout at
`3a67403` was restored and returned `OK`; the final metadata includes that
completion.

## Decode diagnostics and automatic adaptation

A six-request replay at `593ddc9` enabled diagnostic collection while retaining
fixed HOT placement. All JSON values and key order remained correct. Across
the logged windows, HOT coverage averaged approximately 11%, while a
retrospective oracle for each window, using the same capacity, covered
approximately 97%.
CPU and staged arms had similar aggregate cold expert counts, approximately
8.9 per DISK layer and token. Major page-fault rates varied substantially,
including between repetitions of the same mode. PLE gather time stayed near
0.5 milliseconds per decode step in most requests.

These observations point toward cache residency as a useful next target.
They do not isolate a cause: process page faults include more than expert I/O,
window averages are unweighted, and the last tokens can fall outside the
logged windows. Diagnostic timings are excluded from performance claims. The
[replay record](../bench/results/4090-disk-staged-decode-diagnostic-20260905.json)
includes exact request intervals, raw windows, and the parsing script.

Two further starts at `593ddc9` enabled automatic HOT adaptation, with split
histories and phase aim, and disabled diagnostics again. Every measured
prompt received both request orders across starts, with complete warmup
responses in both modes. Each table cell contains four measured responses.

| Workload | CPU wall | Staged wall | Change |
| --- | ---: | ---: | ---: |
| 383-token JSON | 37.173 s | 34.602 s | 6.9% shorter |
| 192-token prose | 24.331 s | 24.203 s | 0.5% shorter |

All twelve JSON responses, including warmups, preserved all 32 records and
key order. JSON improved in three of four measured pairs; prose improved in
two of four. The reversed start had regressions in both workloads. Adaptive
state also carries between alternating modes within a start. This result
does not qualify a general throughput win or establish that adaptation
caused the difference from the earlier fixed-HOT run. The
[automatic-adaptation record](../bench/results/4090-disk-staged-auto-wall-20260905.json)
retains all outputs, journals, and the exact measured client and driver.

## Isolated HOT planning experiment

Staged prefill uses scratch weights, so loading its working set into permanent
HOT slots cannot accelerate that prefill computation. An experiment at
`bfb0d73` instead ranked those slots from decode history during staged chunks,
retaining ordinary phase ranking during CPU fallback. The asynchronous planner
received an immutable preference captured at dispatch. Routes, weights,
arithmetic, observations, token clocks, and swap budgets were unchanged.

Four independent starts at `f03159d` compared the old phase policy with this
decode-focused policy in old/new/new/old order. Each start used one policy
throughout startup, full-response warmup, and measurement. Every request used
staged prefill, so this compares HOT planning policies rather than CPU/GPU
execution. Diagnostic collection, GPU timing, and KV reuse were off. All four
starts retained the same 20 pinned layers, 28 DISK layers, and 14 CPU workers.

| Workload | Phase HOT wall | Decode-focused HOT wall | Change |
| --- | ---: | ---: | ---: |
| 383-token JSON | 40.767 s | 29.856 s | 26.8% shorter, all four pairs improved |
| 192-token prose | 25.174 s | 27.332 s | 8.6% slower, two of four pairs improved |

JSON decode time after the first token fell from 25.486 to 17.579 seconds.
All twelve JSON responses, including warmups, preserved every record, key
order, and key multiplicity. Measured usage counts match. Prose output differs
and remains unscored. Its mixed per-prompt results and slower pooled mean do
not establish a dependable benefit for that workload.

The experiment is retained in the
[complete record](../bench/results/4090-staged-decode-hot-wall-20260905.json),
including the exact runtime patch, driver, client, outputs, and journals.
The automatic decode-history preference is reverted. Avoiding prefill-driven
swaps helped JSON here, but a prefill history can also provide information
about the next response's decode workload. The observed tradeoff needs more
evidence before changing the user's phase policy automatically. Host page
cache was not dropped between starts, so nonlinear residency effects remain
possible even with the reversed start order.

CPU remains the default, and the staged path remains experimental. A fresh
comparison against CPU prefill, with a qualified adaptation policy and whole
responses, is still needed. The original clean serving checkout was restored
and returned `OK` after the fourth start.

## Cache advice review

The existing `HostBank.release_rows()` NOREUSE branch opens a fresh file
descriptor, applies advice, then closes it without reading or mapping from
it. Linux 6.8's [fadvise implementation](https://github.com/torvalds/linux/blob/v6.8/mm/fadvise.c#L108)
sets a flag on that open file description, rather than changing the supplied
page range. The existing mapping therefore does not inherit the advice.
This is a source-level finding; eager release is disabled in these benchmarks
and does not explain their CPU/staged difference.

Applying NOREUSE to the staging descriptor is also not an established remedy:
upstream Linux 6.8's [buffered read path](https://github.com/torvalds/linux/blob/v6.8/mm/filemap.c#L2626)
still marks read folios as accessed, while the
[mapping recency check](https://github.com/torvalds/linux/blob/v6.8/include/linux/mm_inline.h#L580)
consults the flag. Node-4 runs `6.8.0-138-generic`; these upstream sources do not
audit every distribution patch. Any cache-advice or direct-I/O experiment
needs its own whole-response gate, including warm decode after long prefill.

## Correctness validation

At `99bfb4b`, 318 targeted Linux tests passed across staging, materialization,
DISK dispatch, pinned prefill, offload, CLI/configuration, and host budgeting.
Twenty GPU transport/materialization tests also passed CUDA memcheck with
zero reported errors. They cover:

- Exact uint8, float8, and float16 bytes, including unaligned file offsets.
- Repeated small-ring reuse, duplicate/empty/invalid row sets, short reads,
  and EOF errors.
- Selected versus full staging followed by real NVFP4 GEMMs at 16 and 512
  tokens: identical BF16 output bits, with NaNs in unselected scale rows.
- Sparse ownership and protected HOT mappings, with and without overlap.
- CPU fallback across chunk boundaries, scratch reservations, and preserved
  adaptation observations without changing the GPU's routed IDs or weights.

The exploratory change at `bfb0d73` passed 111 targeted Linux tests across HOT
adaptation, diagnostic gating, and DISK prefill policy. Its added cases checked
decode-focused ranking, unchanged explicit blend behavior, immutable planner
selection, and CPU fallback. Those exploratory runtime and test changes are
reverted together; the validation log remains with the experiment. After the
revert at `b1f45f8`, all 100 retained tests in those three files passed again.

## Earlier transport gates

These initial single-start probes explain the chosen transport. They used
only two timing requests per mode and size. Compare modes within each start;
the changing CPU baselines make cross-start comparisons misleading.

| Policy comparison | 524-token TTFT | 2,060-token TTFT |
| --- | ---: | ---: |
| CPU to populate then pageable copy | 10.321 to 18.446 s | 24.431 to 25.284 s |
| CPU to full-row pinned staging | 6.281 to 19.759 s | 24.292 to 22.061 s |
| Full-row to selected-row pinned staging | 14.520 to 7.494 s | 18.385 to 16.977 s |

Populate-read removed the earlier whole-pageable-copy 120-second timeout,
but did not establish a win. Direct full-row staging still moved too many
unneeded experts. Selected staging reduced that traffic while preserving
the GPU computation.

All 42 timing/fidelity pairs across these three early gates matched text and
usage. Seven of the eight fidelity prompts were below the GPU threshold and
therefore used CPU in both arms; only the long retrieval check exercised the
GPU path. The subsequent balanced test above closes that coverage gap.

Raw records: [populate](../bench/results/4090-disk-copy-populate-20260905.json),
[full staging](../bench/results/4090-disk-staging-wall-20260905.json),
[selected staging](../bench/results/4090-disk-selected-staging-20260905.json).
The preceding cache ownership fix is documented in
[disk-copy-prefill.md](disk-copy-prefill.md).
