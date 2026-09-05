# Preserve protected experts during copy prefill

The `--moe-disk-prefill copy` escape hatch materializes each DISK layer into
GPU slots `[0, E)` before running its original routed expert computation.
A HOT layer also has permanent expert copies in protected slots outside the
prefill buffers.

Previously, materialization removed every slot owner belonging to that layer,
including protected owners, and redirected all expert mappings to `[0, E)`.
Reusing the prefill buffer then removed those mappings. Decode still classified
the expert as HOT, allocated an ordinary cache slot for the apparent miss, and
skipped copying its weights because HOT copies are assumed to be permanent.
The selected slot could therefore contain another expert's weights.

HOT-layer materialization now treats `[0, E)` as temporary, unowned copies.
It invalidates displaced ordinary owners while retaining protected mappings
and their permanent usage sentinel. Whole-layer transfers and the GPU GEMM
still use the same rows, weights, scales, and router choices. Ordinary layers
retain their existing materialization and decode-cache behavior. This change
adds no weight buffers and does not enable copy prefill by default.

## GPU validation

On node-4's RTX 4090, the new regression failed before the fix at both
`E=8/S=32` and Qwen's `E=512/S=4045` cache geometry. After materializing HOT
and non-HOT layers and reusing the prefill buffer, decode returned expert 0/1
bytes for a request for expert 1/3.

The fixed regression checks all six NVFP4 banks byte for byte, including
packed weights, block scales, and global scales. It also checks protected
ownership and usage sentinels after repeated materialization, buffer reuse,
and HOT decode remapping. The cases now cover both 10-route scalar decode
and 80-route tiled decode at each cache size. The existing ordinary
materialization bookkeeping regression also passes.

```sh
python -m pytest tests/moe/test_materialize_hot_slots.py \
  tests/moe/test_offload.py::test_nvfp4_materialize_keeps_bookkeeping_consistent_across_requests -q
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/moe/test_materialize_hot_slots.py -q
```

At final test revision `04fa9f8`, all five targeted checks passed. All four
new cases also passed under CUDA memcheck with zero reported errors.

## Bounded wall-time check, 2026-09-05

The existing whole-layer copy policy was tested after the correctness fix on
Qwen Flash NVFP4, using the same server and static HOT placement as its CPU
reference. Runtime revision: `5a09b19`, with CPU input reuse enabled. Startup
allocated 20 pinned layers (26.44 GiB), 28 DISK layers (37.02 GiB), and 82
protected experts per DISK layer. The CPU executor used 14 workers.

Diagnostic collection, GPU step timing, HOT adaptation, HOT plan persistence,
and disk KV reuse were off. The KV cache was naive; no prompt tokens were
reused. io_uring PLE and selective pinned-layer transfers up to 128 tokens
were enabled. A benchmark-only control hook changed the DISK policy and its
pinned-buffer schedule at a prefill boundary. It collected no profiling data.

The 2,060-token prompt generated `OK` in CPU mode:

| Request | Client time to first token | Whole request |
| --- | ---: | ---: |
| CPU warmup | 22.441 s | 22.545 s |
| CPU reference | 19.474 s | 19.551 s |
| Whole-layer copy | No first token within 120.103 s | Timed out |

The server journal records completion of both CPU chunks (2,048 and 12 tokens)
and no completed chunk for the subsequent copy request before cancellation.
This is a failed initial gate, not a completed paired performance study.
Remaining copy repetitions and model-fidelity checks were skipped after the
timeout. There is no model-quality qualification for copy prefill from this
run. The regression independently validates exact expert bytes and ownership.

Keep CPU prefill as the default. Avoiding full copies only for short tail
chunks would not address this observed first-chunk stall. Further GPU work
needs to investigate bounded staging and selected-row transport before
retrying the complete model. This measurement does not isolate disk I/O,
page faults, pageable-copy overhead, or GPU compute as the individual cause.

The [raw record](../bench/results/4090-disk-copy-20260905.json) retains the
request, client results, driver, journal, and before/after regression logs.
