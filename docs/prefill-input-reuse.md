# Reuse CPU prefill input quantization

The native NVFP4 batch path originally gathers one BF16 input row per route,
then quantizes every gathered row to its existing int8 activation format.
A token sent to ten cold experts therefore repeats the same quantization ten
times before the worker pool starts.

The new preparation path walks tokens and their original top-k routes. It
quantizes a token directly into its first valid route's grouped row, then
copies those exact bytes and block scales to its other valid routes. No
additional workspace is allocated. Tokens without valid cold routes do no
input preparation work. Repeated expert IDs remain separate routed rows.

Router weights are still applied after the input projection. Expert grouping,
GEMMs, intermediate rounding, and the final top-k reduction stay unchanged.
The reference gather path remains available through the native executor's
`set_prefill_input_reuse(False)` method for paired tests and benchmarks.
The method samples its switch once per preparation and collects no telemetry.

## Validation plan

Rebuild the CPU extension on Linux, then run:

```sh
python -m pytest tests/moe/test_cpu_moe_prefill_batch.py -q
python bench/cpu-prefill-inputs.py --output /tmp/cpu-prefill-inputs.json
```

The new native parity cases compare raw BF16 output bits with both paths,
including Qwen expert dimensions, router weights on input and output, different
worker counts, duplicate experts, leading absent routes, empty route sets,
changing batch sizes, and reused workspaces. They also check finite results,
unchanged inputs, bounded allocation, and that batch execution stays enabled.

The standalone CPU benchmark alternates both modes on resident synthetic
weights at 64, 512, and 2,048 tokens with one, four, and ten cold routes per
token. Its results isolate CPU work. Model throughput must be assessed with
paired client wall time, fixed placement, no prefix reuse, and diagnostic
collection disabled before making a serving recommendation.

## Native results, 2026-09-05

All 122 focused checks passed on node-4, covering the CPU batch path, DISK
prefill, and composition with the concurrent-prefill prototype. All 63
measured native benchmark pairs produced identical BF16 output bits.

For 2,048 tokens at Qwen's H=2560/I=640 dimensions, median whole-batch CPU
times across seven pairs were:

| Cold routes per token | Reference (ms) | Input reuse (ms) | Time reduction |
| ---: | ---: | ---: | ---: |
| 1 | 83.154 | 82.062 | 1.3% |
| 4 | 275.816 | 219.915 | 20.3% |
| 10 | 677.541 | 526.060 | 22.4% |

Across all tested sizes, multiple cold routes showed 14.2% to 25.6% lower
CPU time. A single cold route changed by only 0.6% to 1.3%. These timings
include preparation, worker GEMMs, and scatter, with 14 workers on the
Ryzen 7 7800X3D and 128 synthetic resident experts. They exclude model
execution, disk reads, and GPU transfers. No workspace growth occurred.

The [raw CPU record](../bench/results/4090-prefill-input-reuse-cpu-20260905.json)
includes every measured call and the test log. Runtime revision: `0059249`.
Non-debug model wall-time validation is pending. This change is not deployed
in the original serving checkout.
