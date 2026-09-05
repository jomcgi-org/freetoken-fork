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

## Direct staging model gate

A second server compared CPU prefill against direct pinned staging of all
expert rows. Both arms allocated the same 64 MiB ring before warmup. The other
flags, geometry, prompts, order, and 512-token threshold were unchanged.
Runtime revision: `632f3f2` plus the recorded hook.

| Prompt tokens | CPU TTFT | Full-row staging TTFT |
| ---: | ---: | ---: |
| 2,060 | 24.292 s | 22.061 s |
| 524 | 6.281 s | 19.759 s |
| 76 | 1.811 s | 1.963 s |

The two long-prompt staging times were 25.227 and 18.894 seconds, against
23.762 and 24.823 seconds on CPU. The mean is lower, but one pair regressed
and the sample does not establish a repeatable win. Medium-prompt staging
was substantially slower. The small workload remains CPU execution in both
arms and is a control, not evidence of a staging effect.

All six timing pairs and eight fidelity pairs matched text and usage. Both
modes retained the same 7/8 answers, with the same caveat that only the long
retrieval case exercised GPU prefill in the fidelity set.

The [populate record](../bench/results/4090-disk-copy-populate-20260905.json)
and [direct staging record](../bench/results/4090-disk-staging-wall-20260905.json)
retain complete drivers, results, and journals. Compare policies within each
server; the different CPU baselines show why comparing across these starts
would be misleading.

## Staging only the original routed expert union

A third server held GPU computation fixed and changed only which rows were
read and staged. Both arms used direct pinned staging for chunks of at least
512 tokens. The selected arm copied exactly the original router's expert
union into the original row positions. It did not substitute hot experts,
change router weights, or change the GEMM. Runtime revision: `632f3f2` plus
the recorded hook.

| Prompt tokens | Full-row staging TTFT | Selected-row staging TTFT |
| ---: | ---: | ---: |
| 2,060 | 18.385 s | 16.977 s |
| 524 | 14.520 s | 7.494 s |
| 76 | 1.818 s | 2.094 s |

Both medium-prompt pairs improved. The long-prompt pairs were mixed:
19.658 to 15.174 seconds, then 17.113 to 18.779 seconds. The small workload
again executes on CPU in both modes. This is a transport comparison, not
proof of a CPU prefill throughput improvement.

All six timing pairs and eight fidelity pairs matched text and usage. The
long retrieval answer was correct in both arms (13.728 versus 9.471 seconds
to first token); the other seven fidelity prompts used CPU in both arms.
The [selected-row record](../bench/results/4090-disk-selected-staging-20260905.json)
contains all results, drivers, and journals.

The final GPU checks at `c5dc111` add full NVFP4 GEMM comparisons at 16 and
512 tokens. Selected staging and full staging produced identical BF16 output
bits, even with NaNs in every unselected scale row. All twelve tests pass,
and all twelve pass CUDA memcheck with zero reported errors.

Before a serving recommendation, compare selected staging directly against
CPU prefill with balanced prompt order across starts. Integrating sparse
copies also requires correct ownership for non-HOT layers: their uncopied
rows must not be advertised as cache hits. These probes used 28 HOT layers,
whose scratch copies are already unowned through the preceding ownership fix.
