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

## Prefill chunk screening, 2026-09-22

The selected profile retains 2048-token chunks. A node-4 screening sweep with
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
