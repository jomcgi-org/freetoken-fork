# Parallel HOT routing on the RTX 4090

HOT/COLD routing preserves the router's selected experts and weights. Resident
experts run on the GPU, and the other selected experts run on the CPU. This
change accelerates that bookkeeping without changing the HOT selection policy,
native NVFP4 weights, scales, activation precision, or expert computation.

Route sets of at least 64 entries use tiled integer histograms and parallel
slot lookups. Tiles have at most 1,024 entries. Smaller route sets retain the
serial algorithm. Set `FREETOKEN_PARALLEL_HOT_ROUTING=0` before starting the
server to compare the synchronized serial path. Correctness fixes remain
enabled in both modes.

## Correctness fixes exposed by measurement

The original in-place remap compiled to redundant scalar loads across warps
with an elected writer. A delayed reader could read an already rewritten cache
slot as an expert ID, causing an out-of-bounds lookup. The original benchmark
repeatedly failed even with the parallel path disabled. Synchronizing readers
before rewriting route IDs resolved those failures in the unchanged harness.
The shared timestamp has the same replicated-reader pattern and is protected
too. These barriers are required in ordinary execution: Triton's
`debug_barrier()` name does not make them optional telemetry.

Split decode also clamps zero-weight cold GPU routes to cache slot 0. When that
slot had never held an expert, allocator residue could produce NaNs before the
zero weight was applied. Initializing only that row when banks are allocated
or rebuilt avoids this without adding per-token work. Tests now deliberately
poison untouched BF16 bank rows with NaNs, so allocator reuse cannot hide the
failure.

## Validation

On node-4's RTX 4090, all 223 focused checks passed:

```sh
python -m pytest \
  tests/moe/test_parallel_hot_routing.py \
  tests/moe/test_hot_adapt.py \
  tests/moe/test_disk_tier.py \
  tests/moe/test_prefill_selective.py -q
```

Coverage includes exact route IDs, cache mappings, victim choices, timestamps,
diagnostic totals, adaptation histories, non-power-of-two expert counts,
all-cold routes, repeated experts, tile tails, production-sized slot tables,
and changing inputs under CUDA graph replay. The poisoned-bank decode test
failed before the slot initialization fix for both initial setup and rebuild.

Compute-sanitizer memcheck reported zero errors for the complete kernel
benchmark after the fixes. Sanitized timings are excluded from performance
results.

## Kernel measurement

```sh
python bench/hot-routing.py --output /tmp/hot-routing-kernels.json
```

The diagnostic benchmark uses 512 experts, 3,753 cache slots, 82 resident
experts, eight warps, adaptation enabled, and diagnostic totals disabled. CUDA
events measure ten reset-and-route operations in a captured graph; the reported
median uses nine replays and includes resetting the original IDs. Both modes
include the correctness barriers.

At 20,480 routes, the synchronized serial path took 1,719.7 microseconds and
parallel routing took 39.3 microseconds, about 44 times faster for this
operation. At ten routes both modes took 3.79 microseconds. These are diagnostic
operation timings, not model throughput claims.

## Non-debug wall-time method

The paired client uses identical prompts, alternating mode order, and one
server to hold expert placement and RAM/VRAM allocation fixed. Disable MoE
statistics, step timing, HOT adaptation, disk KV reuse, and selective prefill
transfers. Use a naive KV cache to prevent prompt-prefix hits. These are
benchmark controls, not production recommendations.

The benchmark server uses a temporary `sitecustomize.py` on its `PYTHONPATH`:

```python
import mmap
import os
from freetoken.moe import offload_kernels
from freetoken.moe.offload_cache import OffloadMoeCache

control_file = open(os.environ["HOT_ROUTING_CONTROL"], "rb")
control = mmap.mmap(control_file.fileno(), 8, access=mmap.ACCESS_READ)
original = OffloadMoeCache.begin_prefill

def begin(self, num_tokens=None):
    offload_kernels._PARALLEL_HOT_ROUTING = bool(int.from_bytes(control[:8], "little"))
    return original(self, num_tokens)

OffloadMoeCache.begin_prefill = begin
```

Create an eight-byte control file before starting that server, then run:

```sh
python bench/hot-routing-wall.py \
  --tokenizer /path/to/flash-e2m1.ftw \
  --control /tmp/hot-routing-control.bin \
  --output /tmp/hot-routing-wall.jsonl
```

The shared mmap read is present in both modes. It selects the prefill policy
and collects no data. There are no server-side GPU timers, diagnostic counters,
or profile polls. Exclude warmups, verify generated text and usage equality,
and compare client first-token and whole-request wall times. Single-token
decode retains the same synchronized scalar route kernel in both modes.
