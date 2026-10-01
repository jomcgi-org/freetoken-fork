"""Per-row FP8 dense linear on the GPU: the Triton GEMV and the dequantize + matmul path against
the pure-PyTorch reference, for the real Qwen3.8-Flash-Next projection shapes.

Shapes are derived from the shipped checkpoint's ``config.json`` (``FREETOKEN_FP8_TEST_MODEL``
overrides the directory). Run on a free GPU only:

    pytest -m cuda tests/kernels/test_fp8_dense_linear_cuda.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

from freetoken.kernel.triton import fp8_dense_linear as fp8
from freetoken.kernel.triton.fp8_dense_linear import (
    candidate_plans,
    clear_tuned_plans,
    fp8_dense_gemv,
    fp8_dense_linear,
    fp8_dense_linear_reference,
    fp8_dequantize,
    make_plan,
    plan_gemv,
    quantize_rowwise_fp8,
    tune_gemv,
)
from freetoken.models.dense_fp8 import projection_shapes

DEV = "cuda"
MODEL_DIR = os.environ.get(
    "FREETOKEN_FP8_TEST_MODEL", "/var/lib/longhorn/nvme-02/freetoken/models/flash-e2m1.ftw"
)


def _shapes() -> dict[str, tuple[int, int]]:
    path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(path):
        pytest.skip(f"no config.json at {path}")
    with open(path) as f:
        return projection_shapes(json.load(f)["text_config"])


def _case(n: int, k: int, m: int, seed: int = 0, dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, device=DEV, generator=g) * 0.03
    w[::7] *= 20.0  # rows of very different magnitude: per-row scales must carry them
    q, scale = quantize_rowwise_fp8(w)
    x = torch.randn(m, k, device=DEV, generator=g).to(dtype)
    return x, q, scale


def _check(y: torch.Tensor, x, q, scale, tol: float = 1e-2) -> None:
    ref = fp8_dense_linear_reference(x, q, scale)
    assert torch.isfinite(y.float()).all()
    err = (y.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
    assert err.item() < tol, err.item()  # bf16 output rounding is ~4e-3 of the max


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "name",
    ["attn.qkv_proj", "attn.o_proj", "gdn.in_proj", "gdn.out_proj", "shared_expert.gate_up_proj",
     "shared_expert.down_proj", "hc.down_block_inject", "hc.up", "hc.mixer_down", "lm_head"],
)
def test_gemv_matches_reference_on_real_shapes(name, m):
    n, k = _shapes()[name]
    x, q, scale = _case(n, k, m, seed=m)
    _check(fp8_dense_gemv(x, q, scale), x, q, scale)
    _check(fp8_dense_linear(x, q, scale), x, q, scale)  # the dispatcher takes the same kernel


@pytest.mark.parametrize("m", [33, 64, 300, 4096])
@pytest.mark.parametrize("name", ["attn.qkv_proj", "gdn.in_proj", "hc.down_block_inject"])
def test_prefill_dequant_path_matches_reference(name, m):
    n, k = _shapes()[name]
    x, q, scale = _case(n, k, m, seed=m)
    _check(fp8_dense_linear(x, q, scale), x, q, scale)


def test_prefill_dequant_path_slices_lm_head_sized_weight():
    n, k = _shapes()["lm_head"]
    x, q, scale = _case(n, k, 48)
    y = fp8_dense_linear(x, q, scale)  # > _DEQUANT_SLICE_ELEMS: several slices, bounded temp
    assert n * k > fp8._DEQUANT_SLICE_ELEMS
    _check(y, x, q, scale)


def test_dequant_kernel_matches_torch():
    n, k = _shapes()["shared_expert.gate_up_proj"]
    _, q, scale = _case(n, k, 1)
    got = fp8_dequantize(q, scale, torch.bfloat16)
    want = (q.float() * scale[:, None]).to(torch.bfloat16)
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_activation_dtypes(dtype):
    n, k = _shapes()["attn.o_proj"]
    x, q, scale = _case(n, k, 3, dtype=dtype)
    y = fp8_dense_linear(x, q, scale)
    assert y.dtype == dtype
    _check(y, x, q, scale)


@pytest.mark.parametrize("n,k", [(17, 48), (100, 1040), (336, 10240), (33, 320)])
def test_ragged_n_and_k_tails(n, k):
    x, q, scale = _case(n, k, 5)
    _check(fp8_dense_gemv(x, q, scale), x, q, scale)


def test_gemv_is_deterministic_run_to_run():
    n, k = _shapes()["hc.down_block_inject"]  # split-K heavy
    x, q, scale = _case(n, k, 1)
    assert plan_gemv(1, n, k).grid_k > 1
    first = fp8_dense_gemv(x, q, scale)
    for _ in range(3):
        assert torch.equal(first, fp8_dense_gemv(x, q, scale))


def test_every_candidate_plan_is_correct():
    n, k = _shapes()["attn.o_proj"]
    x, q, scale = _case(n, k, 1)
    plans = candidate_plans(1, n, k)
    assert plans
    for plan in plans:
        _check(fp8_dense_gemv(x, q, scale, plan), x, q, scale)


def test_gemv_runs_under_cuda_graph_capture_and_replay():
    """Fixed configs, no autotune and no host sync inside the launch, so a captured graph
    reproduces the eager result (incl. the split-K path's scratch allocation)."""
    results = {}
    for name in ("attn.o_proj", "hc.down_block_inject", "gdn.in_proj"):
        n, k = _shapes()[name]
        x, q, scale = _case(n, k, 1)
        eager = fp8_dense_linear(x, q, scale)  # warmup compiles outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fp8_dense_linear(x, q, scale)
        x.copy_(torch.randn_like(x))  # new input, same address
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, fp8_dense_linear(x, q, scale)), name
        results[name] = eager
    assert all(torch.isfinite(r.float()).all() for r in results.values())


def test_tune_gemv_records_valid_plans_and_stays_correct():
    shapes = [_shapes()["attn.o_proj"], _shapes()["hc.down_block_inject"]]
    try:
        best = tune_gemv(shapes, m=1, iters=3, warmup=1)
        assert len(best) == 2
        for n, k in shapes:
            assert plan_gemv(1, n, k) == best[(n, k, 16)]
            x, q, scale = _case(n, k, 1)
            _check(fp8_dense_linear(x, q, scale), x, q, scale)
    finally:
        clear_tuned_plans()


def test_gemv_reads_fewer_bytes_than_bf16_smoke():
    """Not a benchmark (see bench/fp8_dense_gemv.py): only checks the resident weight really is
    half the BF16 bytes plus the row scales."""
    n, k = _shapes()["gdn.in_proj"]
    _, q, scale = _case(n, k, 1)
    assert q.element_size() * q.numel() + scale.element_size() * scale.numel() < 0.51 * 2 * n * k


@pytest.mark.parametrize("n,k,bk", [(320, 320, 256), (100, 1040, 256), (64, 640, 512)])
def test_masked_k_tail_path(n, k, bk):
    """A BLOCK_K that does not divide K takes the masked (EVEN_K=False) variant of the kernel;
    the default planner never picks it for the real shapes, but explicit/tuned plans may."""
    x, q, scale = _case(n, k, 3)
    for split in (1, 3):
        plan = make_plan(3, n, k, block_n=16, block_k=bk, split_k=split)
        _check(fp8_dense_gemv(x, q, scale, plan), x, q, scale)
