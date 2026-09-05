# Bounded file staging for GPU prefill

This experiment investigates the whole-layer copy stall recorded in
[disk-copy-prefill.md](disk-copy-prefill.md). It does not change the serving
policy or enable a new path by default.

## Read preparation before pageable copy

The first probe reused the CPU executor's existing populate-read machinery
before whole-layer GPU copy. It reads every row consumed by the copy through
the existing 32 MiB scratch buffer. A benchmark-only hook selects GPU copy
for chunks of at least 512 tokens and CPU execution for smaller chunks.
The hook also rebuilds the pinned-layer buffer schedule at each policy change.

This eliminated the preceding probe's 120-second first-token timeout. It did
not establish a throughput gain over CPU execution:

| Prompt tokens | CPU TTFT | Populate then copy TTFT |
| ---: | ---: | ---: |
| 2,060 | 24.431 s | 25.284 s |
| 524 | 10.321 s | 18.446 s |
| 76 | 1.790 s | 1.987 s |

Each cell is the mean of two requests in one server. The 76-token workload
uses CPU execution in both modes because it is below the threshold. CPU/copy/
copy/CPU order was used for the long workload; the smaller workloads used
copy/CPU/CPU/copy order. These are initial gates, not a broad performance study.
All six timing pairs returned the same text and usage. Both modes also matched
all eight short fidelity answers, including the baseline's incorrect `108`
answer to the code-trace question. Only the 1,968-token retrieval case is long
enough to exercise GPU copy in that fidelity set; it returned `VIOLET-68243`
in both modes. The other seven cases exercise the unchanged CPU path.

The model, GPU, CPU workers, and memory geometry match the preceding probe:
Qwen Flash NVFP4 on node-4's RTX 4090, 14 CPU workers, 20 pinned expert layers
(26.44 GiB), 28 DISK layers (37.02 GiB), and 82 protected experts per DISK
layer. Diagnostic collection, GPU step timing, HOT adaptation and persistence,
and disk KV reuse were off. The KV cache was naive, with no reused tokens.
io_uring PLE, input reuse, and selective pinned transfers up to 128 tokens
were enabled. Runtime revision: `c2c371b` plus the recorded benchmark hook.

## Direct pinned staging prototype

`DiskPrefillStaging` reads file ranges directly into two reusable pinned
buffers, then copies those unchanged bytes to their original GPU row positions.
The default allocation is 64 MiB total, independent of layer size and request
length. There is no whole-layer host mirror. A completion event protects each
buffer from host reuse until its previous DMA copy finishes. Transfers and
subsequent GEMMs use the caller's current CUDA stream.

The helper supports all rows or an exact selected-row union. It preserves
unaligned file offsets, joins adjacent selected rows, retries partial reads,
and raises on EOF or out-of-range expert IDs. Negative padding IDs are ignored.
Packed weights and scales are transferred as bytes without casts or arithmetic.
The source remains the authoritative file-backed HostBank. UFFD sources are
rejected because their pager owns residency.

Ten targeted GPU tests passed at revision `632f3f2`, including byte parity for
uint8, float8, and float16 banks, unselected-row preservation, repeated reuse
of a small ring, duplicate/empty/invalid row sets, partial reads, and EOF.
All ten also passed CUDA memcheck with zero reported errors.

```sh
python -m pytest tests/moe/test_disk_prefill_staging.py -q
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/moe/test_disk_prefill_staging.py -q
```

The helper is currently used only by a benchmark hook. Serving integration
still needs memory-governor accounting, a measured chunk-size policy, and
quality checks covering the GPU execution path. The owner must synchronize
pending transfers before discarding the staging object.
