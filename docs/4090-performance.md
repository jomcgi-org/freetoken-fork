# Qualified Qwen Flash performance on RTX 4090

The fork includes the ordinary-serving improvements qualified together in #55
and on short coding tasks in #60. Runtime files match the selected build from
#53: CPU prefill input quantization reuse, selective and parallel HOT handling,
protected-slot correctness fixes, buffered staged DISK prefill, published HOT
weight reuse, gated diagnostic work, prefill-marker carry and reclamation of
redundant HOT checkpoint pages. Expert selection and model precision are
unchanged. Every selected expert still contributes to the model result.

On 2026-09-22, node-4 was promoted to merged revision `11654ed`, adding the
harness-root snapshot and tokenizer fixes from #85. Eager and lazy restart
checks reused the shared root, and a matched continuation comparison preserved
request/answer parity across 18 responses with measured means of 74.27 seconds
for the candidate and 74.49 seconds for the previous runtime. See
[disk-prefix qualification details](disk-prefix-cache.md).

The deployment retained 2048-token chunks, the existing native kernels, 100352
reserved KV tokens and the 500 GiB prefix-cache budget. Its health check and
exact-JSON inference smoke check passed. The selected checkout is
`/var/lib/longhorn/nvme-02/freetoken/wt-root-merged-20260922`, selected by
`freetoken-serve.service.d/60-root-runtime.conf`. The prior
`50-selected-runtime.conf` and `wt-astra-qualified-runtime` checkout remain
intact: removing only the 60 override, reloading systemd and restarting the
service restores that configuration. Deployment evidence is under the private
`results/prefill-depth-20260922/root-promotion-*` artifacts on node-4.

This delivers shared-prefix reuse; it does not qualify a larger prefill chunk
or resolve the cold-prefill-to-decode performance tradeoff described below.

Install this revision following the repository's build instructions, including
rebuilding the native CPU extension. An older installed extension does not
contain the selected CPU changes. Use the NVFP4 Qwen Flash checkpoint and its
existing layer profile with the included launch profile:

```sh
bash scripts/serve-qwen-flash-4090.sh /path/to/flash-e2m1.ftw /path/to/layer-profile.json
```

`FREETOKEN_BIN` may name the `ft` executable from the intended Python environment.
Additional arguments are forwarded to `ft serve`, for example
`--served-model-name qwen-flash`. The default alias `qwen3.6-27b` preserves the
existing deployment's client configuration; it does not select a different
checkpoint. The default endpoint is `http://127.0.0.1:8090`.

This is the measured capacity-one configuration: 14 CPU threads, a 6 GiB HOT
budget, 65536 reserved KV tokens, one-row CUDA graphs and FP8 KV. It keeps
ordinary in-memory prefix caching and prefill state carry. Disk prefix
persistence is on by default with a 500 GiB LRU budget under
`FREETOKEN_PREFIX_CACHE_DIR` (`FREETOKEN_PREFIX_CACHE_GIB=0` disables it, and
the directory belongs on local NVMe). Automatic KV growth is disabled. The
cache key covers the checkpoint fingerprint and runtime geometry (dtype, KV
dtype, page size, TP shape), not the serving knobs, so a warmed prefix survives
budget and thread tuning. Different RAM budgets, CPU
thread counts, context capacities and concurrency require separate measurement.
The launch profile makes the selected settings explicit without changing
generic defaults for other models and hardware.

Per-step timing, decode-stat collection, continuation tracing and HOT page
census probes are off. Diagnostic hooks remain available for investigation, but
qualifying wall measurements must run without them. Functional synchronization,
expert adaptation and transfer-lifetime protection remain active. Optional
direct reads and buffered HOT staging remain available but are not selected by
this profile. Experimental decode snapshots are disabled; speculative decoding
and paired CPU kernels are not part of this integrated runtime.

The matched multi-turn comparison ran both execution orders and required the
same requests, complete answers and completion-token counts before accepting a
wall improvement. The short coding comparison retained the original prompts,
tools, budgets, graders and permitted-file checks. These samples establish the
tested behavior, not broad quality equivalence. Detailed measured payloads and
independent audits stay private on the serving node. The prior experimental
branches retain the investigation history; the integrated change contains
runtime source, correctness checks and source-only diagnostic helpers.

Further batching, dense-operation and speculative-verification work is a
separate backlog. Give future experiments a bounded budget and an explicit
workload wall-time target, preserving all task failures and quality checks.
The [model quality checks](model-quality-checks.md) give a minutes-long output
parity gate and an hour-long BFCL tool-calling subset for these experiments.

## Prefill chunk screening, 2026-09-22

Update 2026-09-26: the selected profile now uses 8192-token chunks. On the
deferred-workspace runtime the matched curve cut 100k cold TTFT from 222 s to
89 s with exact parity and no cached-repeat or continuation regression, and a
bounded host-governor charge keeps expert placement unchanged. 16384 does not
fit the 24 GiB card. See the [depth benchmark](prefill-depth-benchmark.md).

The original screening below retained 2048. A node-4 screening sweep with
100352 reserved KV tokens found faster cold prefill at 8192, but slower cached
decode. Both sizes kept 20 PINNED and 28 DISK layers, 6 GiB protected HOT and
the same model/native runtime. Disk-prefix persistence was disabled in every
arm, in-memory radix reuse stayed enabled, and servers restarted between arms.

| Actual prompt tokens | 2048 TTFT | 8192 TTFT | 2048 input tokens / TTFT | 8192 input tokens / TTFT |
| --- | ---: | ---: | ---: | ---: |
| 7959 | 46.61 / 33.61 s | 12.86 s | 171 / 237 tok/s | 619 tok/s |
| 31958 | 82.38 / 96.37 s | 52.30 s | 388 / 332 tok/s | 611 tok/s |
| 99959 | 267.03 s | 167.77 s | 374 tok/s | 596 tok/s |

Slash-separated controls are the first and last observations in the short-depth
sweep, not confidence bounds. The 100k comparison had one observation per arm.
Throughput here is prompt tokens divided by client first-text latency, not a
kernel-only prefill timer. The governor reported zero budget remainder and
charged 0.37 GiB versus 1.18 GiB prefill scratch. No allocation failure bound
the larger setting, but sampled host pressure increased.

At 100k, the ordinary 8192 arm's cached repeat took 8.36 seconds versus 6.30
for 2048. Limiting prefill swaps and enabling a post-prefill adaptation tick
changed the tradeoff without demonstrating preserved decode across the tested
phases. Re-chunking can change floating-point results; these JSON-copy answers
matched exactly, but that is not broad quality equivalence. See the
[full protocol, depth results and continuation checks](prefill-depth-benchmark.md)
for configuration details, fidelity checks and remaining qualification work.

## CPU MoE decode workers, 2026-09-29

The DISK layers' cold experts run on the CPU (W4A8). In isolation the kernel
streams about 55 GB/s on the 7800X3D (one CCD, so the Infinity Fabric link caps
reads near 60 GB/s), the same at 8, 14 or 16 workers and with 4 KB or 2 MB
pages. In serving, 14 workers (8 cores plus 6 SMT siblings) reached only about
13.5 GB/s. `perf` on the worker CPUs showed about 40% of cycles spinning in the
pass barrier, waiting for workers stalled on DISK-bank page faults (about 450
major faults per token). With 8 workers, one per physical core, the kernel ran at
about 21 GB/s and the decode step fell from about 55 to 44 ms.

Outputs are bit-identical across worker counts (1, 2, 8, 14 and 16 at batch 1
and 2). The `threads8q` qualification ran the continuation workload and the
live-like depth rows in ABBA order (14, 8, 8, 14 workers); every arm matched
the parity references.

| Workers | Continuation sessions (s) | 100k cold TTFT | 100k-row decode | Warm decode |
| --- | --- | ---: | ---: | ---: |
| 14 (1st) | 112.1 / 83.2 / 80.5 | 21.69 s | 15.1 tok/s | 22.8 tok/s |
| 8 (2nd) | 95.1 / 64.8 / 62.2 | 21.84 s | 15.6 tok/s | 25.1 tok/s |
| 8 (3rd) | 113.0 / 66.2 / 57.8 | 21.65 s | 18.1 tok/s | 25.9 tok/s |
| 14 (4th) | 103.2 / 70.6 / 66.6 | 21.97 s | 13.6 tok/s | 23.6 tok/s |

Across the bracket, 8 workers cut continuation wall by about 11% (17% on the
warm second and third sessions) and raised decode 10-17%, with prefill
unchanged. The serve script now passes `--moe-cpu-threads 8`. The remaining
CPU-side cost is the page-fault stalls and about 6 ms per step of worker wake
and GIL-bound WILLNEED callback.

## DISK-bank readahead, 2026-09-30

On kernel 6.8 a single `MADV_WILLNEED` reads at most `max(read_ahead_kb,
max_sectors_kb)`, and both were 128 KiB on the model's NVMe. So every decode
WILLNEED for a cold expert row (about 1.6 MiB for gate/up) prefetched only its
first 128 KiB, and the rest of the row arrived one major fault at a time: about
120-140k major faults per 10 s on the worker CPUs, all in the expert banks.
`perf record -e major-faults` put 61% in gate_up, 36% in down and 3% in scales.

Raising `read_ahead_kb` to 2048 on the model drive lets one WILLNEED or fault
cover a whole row. Major faults fell to 1-36k per 10 s. The host sets it with a
udev rule matched to the drive's serial number:

    # /etc/udev/rules.d/60-freetoken-nvme-readahead.rules
    ACTION=="add|change", SUBSYSTEM=="block", KERNEL=="nvme*n1", ATTRS{serial}=="<serial>", ATTR{queue/read_ahead_kb}="2048"

The `raq` qualification ran 8 workers in ABBA order (128, 2048, 2048, 128).
Every arm matched the continuation and depth-row parity references. ABBA means:

| Row | TTFT 128 -> 2048 | Decode 128 -> 2048 |
| --- | ---: | ---: |
| warm decode | 6.23 -> 6.12 s | 23.9 -> 29.2 tok/s |
| 8k cold | 4.36 -> 4.37 s | 17.3 -> 25.2 tok/s |
| 8k repeat | 1.23 -> 1.19 s | 24.3 -> 33.0 tok/s |
| 32k cold | 10.72 -> 10.72 s | 10.9 -> 18.9 tok/s |
| 32k repeat | 1.56 -> 1.90 s | 22.6 -> 20.1 tok/s |
| 100k cold | 21.85 -> 21.83 s | 17.7 -> 19.1 tok/s |

The continuation workload fell from 234.6 to 187.6 s (-20%). The 100k rows
decode only 81 tokens right after a prefill that streams about 34 GiB, so their
decode figure is mostly post-prefill warm-up. The 32k cached repeat regressed
about 11% in both bracketed pairs and remains open.

## KV reservation versus HOT experts, 2026-09-30

The profile reserves 100352 KV tokens at startup. On this model only the 12
full-attention layers (QSA sparse attention) keep per-token KV; the 36 GDN
layers keep a fixed recurrent state. A token costs 12,288 bytes of FP8 K/V plus
768 bytes of compressed index keys, about 12.7 KiB. The whole reservation
is 1.22 GiB of K/V, so shrinking it to 16384 tokens frees about 1.1 GiB, or
395 expert slots (3320 to 3715). The GDN state pool (9 slots, about 1 GiB) does
not scale with context. The largest idle block at decode is the 2.8 GiB
`--memory-ratio 0.87` headroom, which the 64k layer-major prefill needs.

Extra slots only help if the HOT set grows. The 20 GPU-resident layers compute
every expert on the GPU either way, and more LRU slots for them did not change
decode. The `screen1` screen (results in `results/kvstream/`) ran four 1000-token greedy decodes (short
prompts and a 12.5k-token document) per arm, in mirrored order (A B C D D C B A):

| Arm | HOT/layer | Slots | Mean tok/s (pair) | Major faults (pair) |
| --- | ---: | ---: | ---: | ---: |
| 100352 reserved (baseline) | 79 / 82 | 3320 | 21.6 / 26.4 | 1.58M / 0.79M |
| 16384 reserved, HOT 7 GiB | 92 / 96 | 3715 | 24.0 / 28.1 | 1.07M / 0.41M |
| 16384 reserved, HOT 6 GiB | 80 / 80 | 3715 | 21.3 / 26.1 | 1.36M / 0.92M |
| `--kv-ladder on` (floor 65536) | 80 / 80 | 3484 | 24.3 / 22.1 | 0.97M / 1.05M |

Across the bracket, only the larger HOT set gained (+8.6% decode). The static
16384 reservation cannot serve 100k prompts, so it is not a deployable profile.
The page cache warmed during the run, and the DISK layer count moved from 29 to
28 between the first and last arms, so the per-arm figures drift. None of the
1000-token decodes matched between arms, including the two baselines. Which
experts are HOT decides whether an expert runs on the GPU (W4A16) or the CPU
(W4A8), so this screen is not a parity gate.

The existing KV ladder cannot deliver the gain as it stands:

- The floor is at least two 32768-token steps (65536), so it frees only 164
  slots (about 0.4 GiB).
- Growth refuses any rung that would consume protected HOT capacity. The HOT
  budget therefore can only be as large as the capped pool allows, so freed
  slots reach the LRU and never the HOT set.
- Admission plans for input plus `max_tokens`. With the 32768-token default
  output budget, most requests grow the pool right away, and growth is one-way
  until restart.
- Growth is slow. In `growth1`, the ladder (floor 65536, cap 100352) served the
  live-like rows with parity against `workspacecurve1-0`. The first 100k
  prompt waited on a 107 s rebuild: TTFT was 130.1 s, against 22.4 and 22.1 s
  for the next two 100k prompts. The rebuild frees the slot cache and streams all 2296 protected
  HOT rows back from the DISK banks through the 193-row staging ring.

The ladder stays off. Delivering the +8% needs growth that shrinks the HOT set
in place without a full reload (and ideally shrinks back), or KV streaming
like Strata's `--kv-resident`. The streaming option keeps the whole
KV in pinned host memory and a resident window of at least 20480 tokens per QSA
layer in VRAM. Strata reports +6% at 128K and +23% only at 262K. At this
profile's 100k cap the stake is about 1.1 GiB, and the sparse attention would
read up to 2048 selected tokens per layer per step across PCIe.
