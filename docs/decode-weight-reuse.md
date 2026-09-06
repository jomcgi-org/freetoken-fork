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
the same. The native and CUDA checks below validate exact output bits for the
tested geometries and transport paths.

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
Two CUDA graph cases replay fresh, empty, and duplicate routes through both
the host-function and stream-memory-operation transports. Reuse is toggled
between replays of the same captured graph, with exact output-bit comparisons.

The [paired CPU benchmark](../bench/cpu-decode-weight-reuse.py) uses the real
Qwen Flash H=2560/I=640 geometry and fourteen threads by default. It varies batch
size, active routes, and cross-request expert sharing. Each case warms both
modes, reverses their order on alternate repeats, checks exact output bits, and
retains source and native identities. Its synthetic resident weights isolate
native task work; they do not establish end-to-end gains or disk behavior.

Native validation at `b8ac3f7` passes all 69 focused Linux/CUDA checks, including
all 44 new cases without skips. The paired benchmark records 352 exact-output
pairs across 32 cases. Every timed pair has identical BF16 output bits; the
separate direct-kernel tests compare FP32 bits before intermediate rounding.

At batch size four with ten active routes per token:

| Shared experts per token | Existing native task | Reuse native task | Less task time |
| --- | ---: | ---: | ---: |
| None | 1.256 ms | 1.256 ms | Effectively flat |
| Five | 1.235 ms | 0.973 ms | 21.2% |
| Ten | 1.235 ms | 0.694 ms | 43.8% |

These resident-weight measurements isolate native CPU work. Actual expert
sharing, disk faults, GPU execution, and serving adaptation can change the
end-to-end result. A separate non-debug complete-response wall-time and quality
gate remains required before enabling this experiment by default or claiming
a model throughput gain.

The [complete native validation record](../bench/results/4090-decode-weight-reuse-native-20260906.json)
retains sources, native identities, commands, test XML, paired timings, and
service restoration evidence. The first attempt stopped before compilation
because the GPU still reported a process immediately after the service stopped.
Its recovery verified the original service. The second attempt retained that
failure, waited for GPU release, passed compilation/tests/timings, and again
verified a real completion from the restored original service. All native work
started after the concurrent model wall-time driver had exited successfully.

The [model wall-time driver](../bench/decode-weight-reuse-wall-driver.py) freezes
the qualified `b8ac3f7` runtime and native binary. It offers four requests in all
four starts, with reuse off/on/on/off and identical server capacity, KV reserve,
reader policies, and automatic HOT adaptation settings. Each start runs four
warmups, twelve measured complete responses, and eight fidelity cases. It checks
the worker's actual reuse environment value and native mapping before timing.
Only group elapsed time determines aggregate throughput; individual response
latency and all outputs remain in the record.

Launch the detached driver with a two-hour runtime limit and the
[recovery script](../bench/decode-weight-reuse-wall-restore.sh) as `ExecStopPost`.
It waits for GPU release before each benchmark start and restores the original
service with a real completion afterward. `--preflight` checks frozen sources,
the completed native validation record, and binary identity without stopping
the current service. This model comparison remains pending.
