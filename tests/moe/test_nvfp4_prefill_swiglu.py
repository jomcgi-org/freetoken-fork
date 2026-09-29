"""Prefill NVFP4 MoE: the gate/up GEMM with the SwiGLU in its epilogue must produce
every bit the separate GEMM + flashinfer act_and_mul pass produces."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _flashinfer():
    from freetoken.kernel.backend import is_flashinfer_installed

    if not is_flashinfer_installed():
        pytest.skip("the fused epilogue reproduces flashinfer's act_and_mul")


def test_fusion_is_limited_to_silu_without_input_weighting(monkeypatch):
    from freetoken.moe import fused_nvfp4 as F

    monkeypatch.setattr("freetoken.kernel.backend.is_flashinfer_installed", lambda: True)
    assert F._swiglu_fused("silu", False)
    assert not F._swiglu_fused("silu", True)
    assert not F._swiglu_fused("swigluoai", False)
    monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_SWIGLU", "separate")
    assert not F._swiglu_fused("silu", False)
    monkeypatch.delenv("FREETOKEN_NVFP4_PREFILL_SWIGLU")
    monkeypatch.setattr("freetoken.kernel.backend.is_flashinfer_installed", lambda: False)
    assert not F._swiglu_fused("silu", False)


@cuda
def test_epilogue_activation_matches_flashinfer_for_every_bf16_gate():
    """All 65536 bf16 gate values against several up values, bit for bit."""
    _flashinfer()
    import triton
    import triton.language as tl
    from flashinfer import silu_and_mul

    from freetoken.kernel.triton.nvfp4_fused_moe import _silu_mul_fast_math

    @triton.jit
    def apply(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
        tl.store(o_ptr + offs, _silu_mul_fast_math(g, u).to(tl.bfloat16), mask=mask)

    gate = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    for up_value in (1.0, -0.37109375, 3.140625, 1e-3, 250.0):
        up = torch.full_like(gate, up_value)
        up[1::2] = torch.randn(gate.numel() // 2, device="cuda").to(torch.bfloat16)
        reference = silu_and_mul(torch.cat([gate.view(-1, 1024), up.view(-1, 1024)], dim=1))
        out = torch.empty_like(gate)
        apply[(triton.cdiv(gate.numel(), 1024),)](gate, up, out, gate.numel(), BLOCK=1024)
        got, want = out.view(torch.int16), reference.reshape(-1).view(torch.int16)
        # NaN payloads may differ; every other value must match bit for bit.
        nan = torch.isnan(reference.reshape(-1)) & torch.isnan(out)
        mismatch = (got != want) & ~nan
        assert not mismatch.any(), (
            f"{int(mismatch.sum())} mismatches, first gate "
            f"{gate[mismatch.nonzero()[0]].item()} (up {up_value})"
        )


@cuda
@pytest.mark.parametrize("tokens,experts", [(37, 24), (2304, 24), (4096, 512)])
def test_fused_swiglu_is_bit_identical_to_the_separate_pass(monkeypatch, tokens, experts):
    _flashinfer()
    from freetoken.moe import fused_nvfp4 as F

    torch.manual_seed(11)
    hidden, inter, top_k = 2560, 640, 10
    dev = "cuda"

    def bank(n, k):
        return (
            torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=dev),
            (torch.rand(experts, n, k // 16, device=dev) * 0.5 + 0.25).to(torch.float8_e4m3fn),
            (torch.rand(experts, n, device=dev) * 0.01 + 0.001).half(),
        )

    gate_up, down = bank(2 * inter, hidden), bank(hidden, inter)
    weights, ids = torch.topk(
        torch.softmax(torch.randn(tokens, experts, device=dev), -1), top_k, dim=-1
    )
    x = (torch.randn(tokens, hidden, device=dev) * 0.1).to(torch.bfloat16)
    monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_SWIGLU", "fused")
    fused = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_SWIGLU", "separate")
    separate = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    assert torch.equal(fused.view(torch.int16), separate.view(torch.int16))
