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

## Pinned-layer misses on the CPU, 2026-09-30

The 19 pinned layers used to copy every GPU-cache miss over PCIe
(`fast_index_copy_multi`, a UVA gather from the pinned bank): 7.4 ms of a
38.4 ms decode step in an Nsight capture. The existing hybrid backend computes
misses on the CPU instead. The CPU W4A8 executor reads the pinned host bank in
place while the GPU computes its cached experts, then the partials merge
(`_decode_split_partials`, the same path the DISK layers' HOT/COLD split uses).
`--moe-hybrid-max-fetch N` still fetches the N most recently active misses per
layer and step, so the GPU LRU keeps learning. DISK layers, the HOT set,
prefill (including layer-major groups), PLE and CUDA graphs are unchanged.

The CPU pool also serves the DISK layers, but layers run in sequence, so the
pinned layers' CPU work lands in time the pool was idle. The one config change
was letting `--moe-hot-host-cache reclaim` accept `hybrid`. It only touches
DISK HOT rows.

Screens (`--moe-step-timing`, four 1000-token greedy decodes, tok/s):

| Arm | essay1 | essay2 | doc | essay1b | step | CPU compute/step | CPU bytes/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| offload (ABBA mean) | 27.6 | 25.1 | 19.4 | 30.1 | 37.4 ms | 8.6 ms | 362 MB |
| fetch 2 (ABBA mean) | 31.4 | 26.4 | 21.0 | 31.9 | 34.4 ms | 11.0 ms | 490 MB |
| fetch 3 (2 arms) | 30.3 | 24.8 | 20.3 | 30.6 | 35.9 ms | 10.6 ms | 442 MB |
| fetch 1 (sweep) | 31.0 | 24.3 | 19.3 | 29.5 | 36.6 ms | 14.2 ms | 562 MB |
| fetch 0 (sweep) | 26.4 | 21.8 | 18.0 | 25.0 | 41.9 ms | 19.3 ms | 918 MB |

With fetch 0 the LRU never fills and the CPU computes every pinned route. With
fetch 2, Nsight puts the PCIe gather at 3.5 ms per token and the step at
35.2 ms. The `cipq1` qualification (ABBA: offload, hybrid, hybrid, offload):

| Row | offload | hybrid fetch 2 |
| --- | ---: | ---: |
| continuation total (mean) | 190.1 s | 166.8 s |
| warm decode | 32.1 tok/s | 34.5 tok/s |
| 8k cold / repeat decode | 29.6 / 36.0 tok/s | 29.3 / 37.4 tok/s |
| 32k cold / repeat decode | 23.4 / 25.0 tok/s | 20.2 / 23.8 tok/s |
| 100k cold TTFT | 21.8 s | 21.6 s |
| 100k-row decode (81 tokens) | 27.6 tok/s | 27.7 tok/s |

The continuation tasks passed with exact fixed-work parity, and the live-like
texts matched on all rows. The 32k rows decode 81 to 451 tokens right after a
prefill that evicted the LRU. Hybrid refills it at most 2 experts per layer per
step, so these rows are slightly slower. The spread is within the arm-to-arm
noise, but it is the only row that moved the wrong way.

Numerics move from GPU A16 to the CPU W4A8 path for the CPU-computed routes.
Offload itself is not deterministic run to run, because HOT adaptation moves
experts between the GPU and CPU paths. So `bench/parity.py` fails
offload-vs-offload too: 10 of 14 replays failed, and 27% of tokens matched
before the first divergence. Measured against the same offload baseline, on
the positions where neither run had diverged (536 tokens), hybrid matched
offload's own run-to-run drift: top-5 KL 0.0020 against 0.0021, and mean
|dlogprob| 0.018 against 0.018. BFCL subset: offload 41/54 and 34/54 on two
runs, hybrid 36/54 (+2 net against the second offload run).

## Troubleshooting: starts but crawls on long prompts

Symptom: the server starts, answers short prompts at normal speed, then drops to
a few tokens per second on a long one, with the disk busy and the process in `D`
state. `MemAvailable` looks fine and swap is idle.

Cause: the DISK tier and the disk prefix cache read and write through the page
cache. The pinned banks, HOT staging and pager fit under the host ceiling, but
too little file cache is left for the DISK layers' routed experts (and the
prefix cache's restores and fsync'd writes), so every step faults to NVMe. The
server is short of file cache, not of memory.

The startup log shows the reserve it is protecting in the `Host memory budget
table` line (`reserve`, `disk_tier_cache`, `prefix_cache_headroom`). When the
pinned budgets leave less cache than that estimate it logs
`HOST FILE CACHE PRESSURE`. The estimate is a floor: two layers of routed
experts with `--prefill-layer-major-tokens`, every disk layer without it, plus
1 GiB for the PLE table's hot rows and 3 GiB when `--kv-disk-cache-gib` is set.
The `Host memory measured` line reports `VmLck`/`VmPin`/`VmRSS` after the banks
are pinned; `free` understates what locked pages hold.

Knobs: lower `--moe-hot-expert-budget-gib` or `--kv-reserve-tokens` (less pinned
memory), or raise `--host-cache-reserve-gib`. An explicit reserve is never
raised automatically.
