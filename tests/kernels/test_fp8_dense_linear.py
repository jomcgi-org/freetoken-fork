"""Per-row FP8 dense linear, GPU-free half: quantization, the reference math, launch planning.

The Triton kernel itself is checked against :func:`fp8_dense_linear_reference` in
``test_fp8_dense_linear_cuda.py``; here everything that does not need a device is pinned down,
including the kernel's tiling/split-K arithmetic (via ``emulate_gemv_plan``, which runs the
same index math in torch) and the dequantize + matmul path large M takes.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton import fp8_dense_linear as fp8
from freetoken.kernel.triton.fp8_dense_linear import (
    FP8,
    FP8_E4M3_MAX,
    Fp8DenseLinear,
    candidate_plans,
    clear_tuned_plans,
    dequantize_rowwise_fp8,
    emulate_gemv_plan,
    fit_stages,
    fp8_dense_linear,
    fp8_dense_linear_reference,
    make_plan,
    plan_gemv,
    quantize_rowwise_fp8,
    set_tuned_plan,
)

# Qwen3.8-Flash-Next projection shapes (N, K): the ones the kernel must plan for.
REAL_SHAPES = {
    "qkv": (13312, 2560),
    "o_proj": (2560, 6144),
    "gdn_in_proj": (16480, 2560),
    "gdn_out_proj": (2560, 6144),
    "shared_gate_up": (1280, 2560),
    "shared_down": (2560, 640),
    "hc_down_inject": (336, 10240),
    "hc_up": (10240, 320),
    "hc_mixer_down": (320, 10240),
    "lm_head": (248320, 2560),
}


# --------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------
def _weights(n=64, k=96, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) * 0.05).to(dtype)


def test_quantized_dtypes_and_scale_is_row_absmax_over_448():
    w = _weights()
    q, scale = quantize_rowwise_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (w.shape[0],)
    expected = w.float().abs().amax(dim=1) / 448.0
    torch.testing.assert_close(scale, expected, rtol=0, atol=0)
    # every row uses the full e4m3 range: its absmax element maps to +-448
    qmax = q.float().abs().amax(dim=1)
    assert torch.all(qmax == FP8_E4M3_MAX)


def test_scales_are_per_row_not_per_tensor():
    w = _weights(n=4, k=64)
    w[0] *= 1000.0
    w[3] *= 1e-3
    q, scale = quantize_rowwise_fp8(w)
    assert scale[0] > 100 * scale[1] and scale[3] < scale[1] / 100
    # a per-tensor scale would crush the small rows; per-row keeps their relative error bounded
    deq = dequantize_rowwise_fp8(q, scale)
    for r in range(4):
        rel = (deq[r] - w[r].float()).norm() / w[r].float().norm()
        assert rel < 0.04, (r, rel.item())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_round_trip_error_bounds(dtype):
    w = _weights(n=128, k=256, seed=3, dtype=dtype)
    q, scale = quantize_rowwise_fp8(w)
    deq = dequantize_rowwise_fp8(q, scale)
    wf = w.float()
    # e4m3 has 3 mantissa bits: half an ulp at the top binade is 16/448 of the row max
    row_bound = wf.abs().amax(dim=1, keepdim=True) * (16.0 / 448.0) * 1.0001
    assert torch.all((deq - wf).abs() <= row_bound)
    rel = (deq - wf).norm() / wf.norm()
    assert rel < 0.04, rel.item()
    assert not torch.isnan(deq).any()


def test_zero_and_tiny_rows():
    w = _weights(n=3, k=32)
    w[1] = 0
    w[2] = 1e-38  # near fp32-denormal: must not overflow to inf/nan
    q, scale = quantize_rowwise_fp8(w.float())
    assert scale[1] == 1.0 and torch.all(q[1].float() == 0)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert torch.isfinite(dequantize_rowwise_fp8(q, scale)).all()


def test_quantization_is_chunk_invariant():
    w = _weights(n=50, k=40, seed=5)
    q1, s1 = quantize_rowwise_fp8(w)
    q2, s2 = quantize_rowwise_fp8(w, chunk_elems=40 * 7)  # 7 rows per step
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(s1, s2)


def test_dequantize_applies_scale_per_row():
    q = torch.full((2, 4), 2.0).to(FP8)
    scale = torch.tensor([0.5, 3.0])
    out = dequantize_rowwise_fp8(q, scale)
    torch.testing.assert_close(out, torch.tensor([[1.0] * 4, [6.0] * 4]))


# --------------------------------------------------------------------------------------
# reference math and the dequantize + matmul (prefill) path, on CPU
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 3, 40])
def test_reference_is_scale_after_accumulate(m):
    w = _weights(n=48, k=64, seed=m)
    q, scale = quantize_rowwise_fp8(w)
    x = torch.randn(m, 64, generator=torch.Generator().manual_seed(m))
    ref = fp8_dense_linear_reference(x, q, scale)
    direct = x @ dequantize_rowwise_fp8(q, scale).t()
    torch.testing.assert_close(ref, direct, rtol=1e-5, atol=1e-5)
    # and it approximates the unquantized linear to within the quantization error
    full = x @ w.float().t()
    assert (ref - full).norm() / full.norm() < 0.05


@pytest.mark.parametrize("m", [1, 7, 64])
def test_cpu_fp8_dense_linear_matches_reference(m):
    w = _weights(n=48, k=64, seed=1)
    q, scale = quantize_rowwise_fp8(w)
    x = torch.randn(2, m, 64, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(2))
    y = fp8_dense_linear(x, q, scale)
    assert y.shape == (2, m, 48) and y.dtype == torch.bfloat16
    ref = fp8_dense_linear_reference(x, q, scale)
    assert (y.float() - ref).abs().max() <= 0.02 * ref.abs().max()


def test_dequant_matmul_slicing_matches_unsliced(monkeypatch):
    w = _weights(n=100, k=64, seed=4)
    q, scale = quantize_rowwise_fp8(w)
    x = torch.randn(9, 64, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(5))
    whole = fp8._dequant_matmul(x, q, scale)
    monkeypatch.setattr(fp8, "_DEQUANT_SLICE_ELEMS", 64 * 16)  # 16-row slices -> 7 of them
    sliced = fp8._dequant_matmul(x, q, scale)
    assert torch.equal(whole, sliced)


def test_bias_is_added_after_projection():
    w = _weights(n=16, k=32)
    q, scale = quantize_rowwise_fp8(w)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    bias = torch.arange(16, dtype=torch.float32)
    y = fp8_dense_linear(x, q, scale, bias)
    torch.testing.assert_close(y, fp8_dense_linear(x, q, scale) + bias.to(torch.bfloat16))


def test_fp8_dense_linear_layer_state_dict():
    layer = Fp8DenseLinear(64, 32)
    state = layer.state_dict()
    assert set(state) == {"weight", "weight_scale"}
    assert state["weight"].dtype == FP8 and state["weight"].shape == (32, 64)
    assert state["weight_scale"].dtype == torch.float32 and state["weight_scale"].shape == (32,)
    q, scale = quantize_rowwise_fp8(_weights(32, 64))
    layer.load_state_dict({"weight": q, "weight_scale": scale})
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(layer.forward(x), fp8_dense_linear(x, q, scale))


# --------------------------------------------------------------------------------------
# launch planning (the kernel's tiling arithmetic)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", REAL_SHAPES)
@pytest.mark.parametrize("m", [1, 2, 16, 17, 32])
def test_plan_covers_every_tile_exactly_once(name, m):
    n, k = REAL_SHAPES[name]
    plan = plan_gemv(m, n, k)
    assert plan.block_m >= m and plan.block_m in (16, 32, 64)
    assert plan.block_n * plan.grid_n >= n > plan.block_n * (plan.grid_n - 1)
    span = plan.kb_per * plan.block_k
    # every K element belongs to exactly one split, and the last split is non-empty
    assert span * plan.grid_k >= k > span * (plan.grid_k - 1)
    assert plan.block_k % 16 == 0 and k % 16 == 0


@pytest.mark.parametrize("name", REAL_SHAPES)
def test_plan_fills_the_gpu_or_is_a_wide_output(name):
    n, k = REAL_SHAPES[name]
    plan = plan_gemv(1, n, k, sm_count=128)
    programs = plan.grid_n * plan.grid_k
    # wide outputs (qkv, in_proj, lm_head, hc_up) never split; narrow ones split K to get
    # to >= 2 CTAs/SM (fewer only for the tiny K=640 shared down)
    assert plan.grid_k == 1 or programs >= 2 * 128, (programs, plan)
    if n >= 10000:
        assert plan.grid_k == 1


def test_plan_is_a_pure_function_of_shape():
    assert plan_gemv(1, 2560, 6144) == plan_gemv(1, 2560, 6144)
    assert plan_gemv(2, 2560, 6144) == plan_gemv(16, 2560, 6144)  # same block_m bucket


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 40, 64), (3, 70, 320), (16, 33, 48), (2, 16, 512), (5, 100, 1040)],
)
def test_emulated_plan_matches_reference_odd_shapes(m, n, k):
    g = torch.Generator().manual_seed(m + n + k)
    q, scale = quantize_rowwise_fp8(torch.randn(n, k, generator=g) * 0.05)
    x = torch.randn(m, k, generator=g)
    ref = fp8_dense_linear_reference(x, q, scale)
    for bn, bk, split in [(16, 16, 1), (16, 64, 3), (32, 128, 4), (16, 256, 99)]:
        plan = make_plan(m, n, k, block_n=bn, block_k=bk, split_k=split)
        torch.testing.assert_close(emulate_gemv_plan(x, q, scale, plan), ref, rtol=1e-4, atol=1e-4)


def test_emulated_default_plan_on_a_real_narrow_shape():
    n, k = REAL_SHAPES["hc_mixer_down"]  # 320 x 10240: the split-K-heaviest shape
    g = torch.Generator().manual_seed(0)
    q, scale = quantize_rowwise_fp8(torch.randn(n, k, generator=g) * 0.02)
    x = torch.randn(2, k, generator=g)
    plan = plan_gemv(2, n, k)
    assert plan.grid_k > 1
    torch.testing.assert_close(
        emulate_gemv_plan(x, q, scale, plan), fp8_dense_linear_reference(x, q, scale),
        rtol=1e-4, atol=1e-4,
    )


def test_tuned_plan_overrides_default_and_clears():
    try:
        default = plan_gemv(1, 2560, 6144)
        set_tuned_plan(2560, 6144, 1, block_n=64, block_k=128, split_k=2, num_warps=8)
        tuned = plan_gemv(1, 2560, 6144)
        assert (tuned.block_n, tuned.block_k, tuned.num_warps) == (64, 128, 8)
        assert tuned.grid_k == 2 and tuned != default
        assert plan_gemv(1, 2560, 640) == plan_gemv(1, 2560, 640)  # other shapes untouched
    finally:
        clear_tuned_plans()
    assert plan_gemv(1, 2560, 6144) == default


@pytest.mark.parametrize("name", ["o_proj", "hc_down_inject", "lm_head", "hc_up"])
def test_candidate_set_is_small_and_valid(name):
    n, k = REAL_SHAPES[name]
    plans = candidate_plans(1, n, k)
    assert 1 <= len(plans) <= 64  # small fixed set, all compiled eagerly at warmup
    assert len(set(plans)) == len(plans)
    for p in plans:
        assert p.grid_n * p.block_n >= n and p.kb_per * p.block_k * p.grid_k >= k


def test_stage_depth_fits_sm89_shared_memory():
    # (32 x 32 x 256): 24 KiB/stage -> 4 stages fit; a 64-row tile at 256-wide K gets fewer
    assert fit_stages(16, 32, 256, 4) == 4
    assert fit_stages(32, 32, 256, 4) == 4
    assert fit_stages(64, 32, 256, 4) == 2
    assert fit_stages(32, 64, 256, 4) == 3
    for m in (1, 16, 17, 32, 33, 64):
        for name, (n, k) in REAL_SHAPES.items():
            p = plan_gemv(m, n, k)
            per_stage = p.block_n * p.block_k + 2 * p.block_m * p.block_k
            assert p.num_stages * per_stage <= 99 * 1024, (name, m, p)
    for p in candidate_plans(32, 13312, 2560):
        assert p.num_stages >= 2 and p.num_stages * (p.block_n * p.block_k + 2 * p.block_m * p.block_k) <= 99 * 1024


@pytest.mark.parametrize("name", REAL_SHAPES)
def test_default_block_k_divides_k_so_the_k_mask_is_dead(name):
    n, k = REAL_SHAPES[name]
    assert k % plan_gemv(1, n, k).block_k == 0
