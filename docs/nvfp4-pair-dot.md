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
unsupported formats and ISAs. Pair dispatch defaults off. Serving can opt in with
`--moe-cpu-nvfp4-pair on`, which configures the fresh executor before callbacks or
tasks exist and logs the effective setting once at startup. Default-off startup
works with older extensions that lack the setter. Enabled CPU execution rejects
non-NVFP4 formats, missing setters and unsupported ISAs. The setting applies to
CPU expert tasks, including offload configurations with CPU or DISK layers; it
does not enable CPU execution for a GPU-only configuration.

An enabled grouped decode task pairs routes within each
expert, preserves the existing route quantization and activation, and leaves the
final top-k reduction unchanged. Unpaired routes use the ordinary dot. Single-token
tasks retain the ordinary route loop even when the setting is enabled.
The existing batched prefill dot uses a different reduction order and is not the
exact reference.

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
requires representative task validation and separate non-debug wall measurements.
No serving throughput gain is claimed by this experiment.

The focused Linux checks pass for both the dot and complete expert schedules.
Both cost probes completed in both execution orders with exact outputs, followed
by verified original-serving recovery. Native checks skip in the Mac environment
where Torch is absent.

The companion full-model target diagnostic in PR #62 also completed with exact
logits, greedy tokens, committed state, retained rejection prefixes and subsequent
ordinary decode. Its latest source, including the single-token bypass, passed
202 focused Linux checks and the exclusive CUDA rollback check. Original serving
was restored and verified after each model experiment.

Model component timings remain sensitive to execution order and the ordinary
decode controls vary. These checks establish correctness for the tested windows,
not a stable serving gain. The explicit pair setting remains disabled by default.
The startup integration
subsequently passed 402 focused Linux checks, all three exclusive CUDA checks and
twelve real serving fixtures exactly in content, reasoning output, finish reason
and completion-token count. The native setting was enabled before graph capture;
speculative windows and host stop rollback were exercised. Independent checks
verified the saved answers, effective startup setting and original-serving
recovery with a completion.

Separate non-debug serving qualification completed both execution orders with
speculation enabled in both arms. Pairing was the only changed startup setting;
source, native extensions, CPU workspace, cache geometry and request bodies were
fixed. All complete answers and completion-token counts matched, and every
conversation passed independent checks. Repetition favored pairing in both
orders. Combined multi-turn wall time was effectively unchanged and slightly
favored pairing off, so this does not establish a general serving gain. Original
serving recovered with a verified completion after each run. Pairing remains
disabled by default and unselected. Ordinary concurrent decoding with pairing is
an untested follow-up candidate. Detailed timing payloads stay private.
