# LayerScale ideas for the 4090 workload

Reviewed 2026-09-06 against the public documentation and the completed Pi
comparison. The resident-populate experiment completed separately with its
runtime frozen throughout.

LayerScale's [session API](https://docs.layerscale.ai/sessions/) keeps token
history and KV across turns, ingests data ahead of queries, and prepares known
tool-result framing while tools execute. This treats a conversation as continuing
execution. The documentation does not establish an additional hidden model state
that removes computation on unknown future inputs.

For a standard causal transformer, previous-token KV plus token and position
metadata already permits incremental continuation. Resending the same history
need not recompute it: [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
explicitly supports reuse across conversation turns. The
[manifesto](https://layerscale.ai/manifesto) describes the broader goal of doing
work before a query arrives; its claim that existing engines always rebuild all
context is too broad.

The [published benchmark](https://layerscale.ai/benchmarks) is unusually useful
here because it compares LayerScale sessions with its own ordinary API with
prefix caching enabled. At ten turns and an 8192-token prefix, Qwen3-8B-FP8 takes
1711.025 ms in a session versus 1706.753 ms when history is resent. Its Flash
Query headline instead retrieves a previously computed answer. Those are
different measurements. The run uses two H100 80GB GPUs with TP2, repeated
workloads and twenty measured iterations after three warmups. Tool-loop timing
does not score whether useful tool work was completed. It does not establish
4090 performance with experts spilling through RAM and NVMe.

The [configuration reference](https://docs.layerscale.ai/configuration/) documents
continuous admission, a 1024-token forward budget, prefix reuse and prompt-lookup
speculation enabled by default for greedy decode. The draft comes from earlier
context and is target-verified; a flag disables speculation for reproducibility.
No verifier implementation or complete per-engine speculation settings were
inspected. The [model reference](https://docs.layerscale.ai/models/) lists NVFP4
loading, while the [current getting-started requirements](https://docs.layerscale.ai/)
name Hopper and Blackwell. These pages do not qualify our Ada 4090, FTW checkpoint
or three-tier execution. A new model integration claim is not that qualification.

## What FreeToken already retains

Our Qwen Flash has 36 GDN layers and 12 QSA layers. Continuation needs recurrent
and convolution state, PLE state, and QSA state as well as ordinary KV. FreeToken
already implements this in [the hybrid prefix cache](../python/freetoken/kvcache/hybrid_radix_cache.py)
and [disk prefix serialization](disk-prefix-cache.md). The completed Pi gate
enabled RAM prefix reuse on both runtimes. A session identifier alone therefore
does not supply a new inference optimization.

There is a concrete gap worth measuring in
[finish-time cache publication](../python/freetoken/scheduler/cache.py): the live
recurrent state is donated only when its token boundary is page aligned.
Otherwise the next request may resume from the older prefill snapshot. A
tool-call anchor can preserve another boundary, but it also requires alignment.
The Pi gate left `--enable-special-token-ckpt` disabled. This existing option
protects the state just after the tool-call opener; rewriting the call body
still invalidates continuation beyond the first changed token.
The latest recurrent state cannot be attached to an earlier token boundary:
that would change the model's continuation.

The [retrospective Pi usage audit](../bench/results/4090-pi-continuation-audit-20260906.json)
screens the existing records without new inference. In the four measured optimized
tasks, 56 of 63 adjacent call transitions reuse exactly the preceding prompt's
64-token-aligned boundary. Baseline has 74 of 81. One optimized call generates
770 tokens; the next reports 2112 cached tokens and 834 uncached tokens after a
2153-token preceding prompt. This suggests replay beyond the new tool result.

These are length observations, not verified matching token prefixes or measured
avoidable time. Parsed tool arguments can be reserialized by the client and
template. Retaining state after a rewritten token would be incorrect. The timing
run deliberately omitted raw token tracing, so it cannot resolve this ambiguity.
The audit retains source hashes, every transition, warmups separately and its
analysis source. It does not assign a speedup to continuation caching.

## Next bounded experiments

1. In a separate diagnostic run, compare exact generated token prefixes with
   the next rendered request, the deepest valid KV match and the deepest live
   recurrent snapshot. Keep tracing off for the eventual wall-time comparison.
2. If token identity permits deeper reuse, preserve a later aligned decode
   snapshot or a bounded exact continuation state. Preserve KV ownership, GDN,
   PLE and QSA state together, and retain the normal path for edited histories.
   Measure complete Pi tasks including snapshot and restoration costs.
3. Evaluate prompt-lookup drafting independently. It avoids the missing MTP
   head and its resident weight cost, but still needs target verification,
   rejection rollback, EOS handling and preserved sampling semantics. Existing
   greedy verification helpers are a starting point, not a completed integration.
4. Consider early prefill only for exact known tokens. Count background work,
   extra storage traffic and interference with expert residency inside the
   complete task measurement. Do not truncate context or return a confidence-based
   early answer as part of the quality-preserving path.

FreeToken already has chunked prefill and concurrent scheduling. Increasing
offered concurrency from one to four did not give a dependable gain in our
completed fixed-capacity comparison. The 128-stream H100 setup does not supply
an appropriate default for a single agent on node-4.
