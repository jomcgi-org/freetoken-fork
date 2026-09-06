# NVFP4 decode weight unpacking experiment

The persistent CPU decode schedule already groups routes by expert and output
tile. It applies the existing GEMV separately to every route in each group,
which repeats packed-weight decoding and scale lookup when several requests
select the same expert. This experiment shares those operations across at
most four routes. It reads existing quantized activation rows through pointers,
without a new weight buffer, activation repacking, or additional quantization.

The AVX-512 VNNI kernel retains the serial decode kernel's four accumulator
chains, vector reduction, scalar remainder, final global-scale multiplication,
and prefetch distance. The existing prefill batch kernel scales and reduces in
a different order, so it is not used for this path. Gate/up activation handling,
per-route intermediate rounding, and the final original top-k reduction remain
the same. These are implementation intentions pending native bitwise validation.

The experiment defaults off. Set
`FREETOKEN_CPU_MOE_DECODE_WEIGHT_REUSE=1` before constructing the native executor
to enable it. The native `set_decode_weight_reuse` method supports controlled
paired measurements. The setting is sampled once per task before workers wake.
NVFP4 with AVX-512 VNNI can use the path; other formats and ISA paths retain their
existing implementation. Single-route expert groups use the existing GEMV.
Larger groups are partitioned into groups of at most four, with a serial tail
when only one route remains.

The [targeted tests](../tests/moe/test_cpu_moe_decode_weight_reuse.py) compare
FP32 bits directly against serial decode before BF16 rounding can hide a
difference. They cover accumulator and scalar tails, reversed activation-row
pointers, zero scales, full persistent-task output parity, changing batch sizes,
duplicate routes, absent experts, empty tasks, router weights on input/output,
activation variants, and the disabled-VNNI fallback. They also check that model
weights, input activations, route IDs, and route weights remain unchanged.

The [paired CPU benchmark](../bench/cpu-decode-weight-reuse.py) uses the real
Qwen Flash H=2560/I=640 geometry and fourteen threads by default. It varies batch
size, active routes, and cross-request expert sharing. Each case warms both
modes, reverses their order on alternate repeats, checks exact output bits, and
retains source and native identities. Its synthetic resident weights isolate
native task work; they do not establish end-to-end gains or disk behavior.

Native compilation, the targeted Linux tests, and paired performance validation
are pending. They must wait until the live `astra-concurrent-wall-driver` unit
is terminal so they do not interfere with the complete-response comparison.
Even a passing native result requires a separate non-debug model wall-time and
quality gate before enabling this experiment by default or claiming a gain.
