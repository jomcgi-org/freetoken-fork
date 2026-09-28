"""Prefill NVFP4 MoE tile selection: faster tiles must not change any output bit."""

import pytest
import torch


def test_large_tiles_apply_only_to_tuned_shapes_and_large_batches():
    from freetoken.moe.fused_nvfp4 import _prefill_config

    large = _prefill_config(4096, 1280, 2560)
    assert (large["BLOCK_SIZE_M"], large["BLOCK_SIZE_N"], large["BLOCK_SIZE_KB"]) == (64, 128, 32)
    assert _prefill_config(4096, 1280, 2560) == _prefill_config(4096, 2560, 640)
    # Small batches and other models keep the previous tiles.
    assert _prefill_config(1024, 1280, 2560) == _prefill_config(1024)
    assert _prefill_config(4096, 1536, 3072) == _prefill_config(4096)
    assert _prefill_config(32)["BLOCK_SIZE_M"] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_large_tiles_are_bit_identical_to_default_tiles(monkeypatch):
    from freetoken.moe import fused_nvfp4 as F

    torch.manual_seed(3)
    experts, hidden, inter, top_k, tokens = 32, 2560, 640, 10, 2304
    dev = "cuda"

    def bank(n, k):
        return (
            torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=dev),
            (torch.rand(experts, n, k // 16, device=dev) * 0.5 + 0.25).to(torch.float8_e4m3fn),
            (torch.rand(experts, n, device=dev) * 0.01 + 0.001).half(),
        )

    gate_up, down = bank(2 * inter, hidden), bank(hidden, inter)
    logits = torch.randn(tokens, experts, device=dev)
    weights, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    x = (torch.randn(tokens, hidden, device=dev) * 0.1).to(torch.bfloat16)
    tuned = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    default = F._prefill_config
    monkeypatch.setattr(F, "_prefill_config", lambda m, n=None, k=None: default(m))
    reference = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    assert torch.equal(tuned, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [100, 2304])
def test_v2_prefill_kernel_is_bit_identical_to_v1(monkeypatch, tokens):
    from freetoken.moe import fused_nvfp4 as F

    torch.manual_seed(5)
    experts, hidden, inter, top_k = 24, 2560, 640, 10
    dev = "cuda"

    def bank(n, k):
        return (
            torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=dev),
            (torch.rand(experts, n, k // 16, device=dev) * 0.5 + 0.25).to(torch.float8_e4m3fn),
            (torch.rand(experts, n, device=dev) * 0.01 + 0.001).half(),
        )

    gate_up, down = bank(2 * inter, hidden), bank(hidden, inter)
    weights, ids = torch.topk(torch.softmax(torch.randn(tokens, experts, device=dev), -1), top_k, dim=-1)
    x = (torch.randn(tokens, hidden, device=dev) * 0.1).to(torch.bfloat16)
    monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_KERNEL", "v2")
    v2 = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    monkeypatch.setenv("FREETOKEN_NVFP4_PREFILL_KERNEL", "v1")
    v1 = F.fused_experts_nvfp4(x, *gate_up, *down, weights.float(), ids.int().clone(), num_experts=experts)
    assert torch.equal(v2, v1)
