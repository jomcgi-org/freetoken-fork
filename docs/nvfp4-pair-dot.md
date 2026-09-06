# Exact two-input NVFP4 dot experiment

The grouped CPU decode schedule visits every route selecting an expert, but each
route independently unpacks the same weight rows. The candidate AVX-512 VNNI
kernel unpacks each weight group once for two inputs. Each input keeps the
ordinary decode kernel's four FP32 accumulation chains, group scale order,
horizontal reduction, scalar tail and final global scale multiplication.
Activation quantization and expert selection are unchanged.

The candidate is available through the explicit `nvfp4_pair_dot_probe`
diagnostic entry point and `CpuMoeExecutor.set_nvfp4_pair_dot`. Configure the
executor before submitting tasks, with no task in flight. The setter rejects
unsupported formats and ISAs. Pair dispatch defaults off and the serving startup
path does not enable it. An enabled grouped decode task pairs routes within each
expert, preserves the existing route quantization and activation, and leaves the
final top-k reduction unchanged. Unpaired routes use the ordinary dot.
The existing batched
prefill dot uses a different reduction order and is not the exact reference.

`tests/moe/test_nvfp4_pair_dot.py` compares FP32 bit patterns against two ordinary
decode dots. Cases cover vector and scalar tails, both model inner dimensions,
finite scale encodings, zero inputs, signs and extreme valid int8 activations.
Complete expert-output comparisons additionally cover overlapping, disjoint,
duplicate and invalid routes, input/output router weighting, activation clamps,
odd route counts and the model's expert dimensions. Native checks require Linux
with AVX-512 VNNI.

`bench/nvfp4-pair-dot.py --output /private/pair-dot.json` pins one CPU and measures
both kernels in both orders. Inputs include small row tiles, expert matrices and
a weight pool larger than LLC. Run it with exclusive benchmark ownership and
automatic original-serving recovery, after builds and validation finish. It
writes detailed results only to the requested private path.

`bench/nvfp4-pair-executor.py --output /private/pair-executor.json` measures the
complete CPU expert layer with one and multiple workers, overlapping and disjoint
routes, and odd batch sizes. Input copying is outside the timer; task submission,
activation preparation, both projections and ordered route reduction are inside.
The same exclusive ownership and recovery requirements apply.

Kernel parity and component cost are preliminary gates. A serving change still
requires full expert-output parity, model verification and separate non-debug
wall measurements. No serving throughput gain is claimed by this experiment.

The focused Linux checks pass for both the dot and complete expert schedules.
Both cost probes completed in both execution orders with exact outputs, followed
by verified original-serving recovery. Native checks skip in the Mac environment
where Torch is absent. Full-model verification and serving wall qualification
remain pending, and the explicit pair setting remains disabled at startup.
Detailed timing payloads stay private.
