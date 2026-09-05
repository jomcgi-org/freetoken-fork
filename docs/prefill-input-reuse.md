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

Native validation and wall-time measurements are pending. This change is not
deployed in the original serving checkout.
