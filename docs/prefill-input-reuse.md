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
python -m pytest tests/moe/test_cpu_moe_prefill_batch.py tests/moe/test_disk_tier.py -q
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

All 105 focused checks passed on the final independent branch on node-4,
covering the CPU batch path and DISK prefill. All 63
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
includes every measured call and the validation logs. CPU benchmark runtime
revision: `0059249`; final branch validation: `a7492fe`.

## Model wall time

Four model starts used the automatic RAM budget, 20 pinned layers, 28 DISK
layers, 82 protected experts per DISK layer, 14 CPU workers, io_uring PLE,
and selective transfers up to 128 tokens. Concurrent prefill, HOT adaptation,
HOT plan persistence, disk KV reuse, diagnostic collection, and GPU timing
were off. The KV cache was naive. Every request used greedy generation with
thinking disabled.

Each start ran four pairs per prefill size and two 192-token decode pairs,
plus warmups and fidelity checks. First/second position was balanced within
each workload. The last two starts reversed every individual prompt's order
with `--order-offset 1`, giving each prompt equal counts of both orders.

| Prompt tokens | Reference TTFT (s) | Input reuse TTFT (s) | Reduction |
| ---: | ---: | ---: | ---: |
| 76 | 1.886 | 1.788 | 5.2% |
| 524 | 5.936 | 5.248 | 11.6% |
| 2,060 | 18.771 | 15.652 | 16.6% |

Every one of the 48 measured prefill pairs was faster with reuse. Each table
cell contains 16 requests across four starts. Whole-request reductions were
5.0%, 11.4%, and 16.6%. All 56 timing pairs, including decode, produced
identical text and usage. Each mode retained the same 7/8 fidelity score and
matched all eight reference answers on every start, including the known
baseline `108` answer to the code question whose correct result is `68`.

The initial two starts appeared to slow decode because one essay had a much
larger first-occurrence cache cost, and reuse always received that position.
Reversing the order reversed the apparent slowdown. Pooling both orders gave
21.78 versus 22.22 tokens/s after the first token and whole-request means of
10.339 versus 10.107 seconds. Decode code is unchanged; this sample does not
establish a decode improvement.

The [complete model record](../bench/results/4090-prefill-input-reuse-wall-20260905.json)
retains all four starts, including the initial apparent decode regression,
exact drivers, startup geometry, outputs, and final validation. Model
measurements used `a6a07aa` and `550d408` with the overlap prototype disabled.
The final PR depends directly on the HOT-routing branch; its native kernel
source is identical to the measured source.

The original clean serving checkout at `3a67403` was restored and returned
a real `OK` completion. This change is not deployed there.
