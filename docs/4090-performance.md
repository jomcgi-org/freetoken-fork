# Qualified Qwen Flash performance on RTX 4090

The fork includes the ordinary-serving improvements qualified together in #55
and on short coding tasks in #60. Runtime files match the selected build from
#53: CPU prefill input quantization reuse, selective and parallel HOT handling,
protected-slot correctness fixes, buffered staged DISK prefill, published HOT
weight reuse, gated diagnostic work, prefill-marker carry and reclamation of
redundant HOT checkpoint pages. Expert selection and model precision are
unchanged. Every selected expert still contributes to the model result.

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
persistence and automatic KV growth are disabled. Different RAM budgets, CPU
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
