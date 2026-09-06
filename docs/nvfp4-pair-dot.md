# Exact two-input NVFP4 dot experiment

The grouped CPU decode schedule visits every route selecting an expert, but each
route independently unpacks the same weight rows. The candidate AVX-512 VNNI
kernel unpacks each weight group once for two inputs. Each input keeps the
ordinary decode kernel's four FP32 accumulation chains, group scale order,
horizontal reduction, scalar tail and final global scale multiplication.
Activation quantization and expert selection are unchanged.

The candidate is available only through the explicit `nvfp4_pair_dot_probe`
diagnostic entry point. Serving dispatch is unchanged. The existing batched
prefill dot uses a different reduction order and is not the exact reference.

`tests/moe/test_nvfp4_pair_dot.py` compares FP32 bit patterns against two ordinary
decode dots. Cases cover vector and scalar tails, both model inner dimensions,
finite scale encodings, zero inputs, signs and extreme valid int8 activations.
The native tests require Linux with AVX-512 VNNI.

`bench/nvfp4-pair-dot.py --output /private/pair-dot.json` pins one CPU and measures
both kernels in both orders. Inputs include small row tiles, expert matrices and
a weight pool larger than LLC. Run it with exclusive benchmark ownership and
automatic original-serving recovery, after builds and validation finish. It
writes detailed results only to the requested private path.

Kernel parity and component cost are preliminary gates. A serving change still
requires full expert-output parity, model verification and separate non-debug
wall measurements. No serving throughput gain is claimed by this experiment.
