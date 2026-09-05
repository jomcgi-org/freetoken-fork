"""Selective DMA must preserve expert bytes, kernels, and buffer ownership."""

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(int(v) for v in (torch.version.cuda or "0.0").split(".")[:2]) < (13, 0),
    reason="requires CUDA 13 batch memcpy",
)


def _cache(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS", "512")
    # Real Qwen expert geometry exercises partial copies of the packed banks
    # and full copies of the smaller per-block/per-channel scale banks.
    e, h, i, layers = 8, 2560, 640, 3
    rng = torch.Generator().manual_seed(417)
    sources = {}
    for name, shape, dtype in (
        ("gate_up_packed", (e, 2 * i, h // 2), torch.uint8),
        ("gate_up_scale", (e, 2 * i, h // 16), torch.float8_e4m3fn),
        ("gate_up_global", (e, 2 * i), torch.float16),
        ("down_packed", (e, h, i // 2), torch.uint8),
        ("down_scale", (e, h, i // 16), torch.float8_e4m3fn),
        ("down_global", (e, h), torch.float16),
    ):
        sources[name] = [
            (
                torch.randint(0, 256, shape, dtype=dtype, generator=rng)
                if dtype == torch.uint8
                else (torch.rand(shape, generator=rng) * 0.025 + 0.025).to(dtype)
            ).pin_memory()
            for _ in range(layers)
        ]
    cache = OffloadMoeCache(
        num_layers=layers, num_experts=e, cache_size=3 * e,
        device=torch.device("cuda"), quant_format="nvfp4", prefill_overlap=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


@pytest.mark.parametrize("rows", [[2], [0, 2, 3], list(range(8))])
def test_selected_bytes_and_unselected_poison_across_buffer_reuse(monkeypatch, rows):
    cache, sources = _cache(monkeypatch)
    for _ in range(2):
        for bank in cache.bank_caches.values():
            bank.view(torch.uint8).fill_(255)
        cache.begin_prefill(31)
        assert cache.prefill_selective_active
        for layer in range(3):
            ids = torch.tensor([rows, rows], device="cuda", dtype=torch.int32)
            before = cache.prefill_h2d_bytes
            cache.prefetch_routed_prefill_layer(layer, ids)
            views = cache.wait_prefill_layer(layer)
            for view, (name, per_layer) in zip(views, sources.items()):
                assert torch.equal(
                    view.view(torch.uint8)[rows].cpu(),
                    per_layer[layer].view(torch.uint8)[rows],
                ), (layer, name)
            moved = cache.prefill_h2d_bytes - before
            full = sum(bank[layer].numel() * bank[layer].element_size() for bank in sources.values())
            if len(rows) == 8:
                assert moved == full
            else:
                packed = sum(
                    bank[layer].numel() * bank[layer].element_size()
                    for name, bank in sources.items() if name.endswith("packed")
                )
                assert moved == full - packed + packed * len(rows) // 8
                untouched = sorted(set(range(8)) - set(rows))
                for view, name in zip(views, sources):
                    if name.endswith("packed"):
                        assert (view.view(torch.uint8)[untouched] == 255).all()
            cache.release_prefill_layer(layer)
        # A large following chunk restores lookahead and full-layer copies.
        cache.begin_prefill(2048)
        assert not cache.prefill_selective_active
        cache.prefetch_prefill_layer(0)
        for view, source in zip(cache.wait_prefill_layer(0), sources.values()):
            assert torch.equal(view.view(torch.uint8).cpu(), source[0].view(torch.uint8))
        cache.release_prefill_layer(0)


@pytest.mark.parametrize("tokens", [1, 17, 129])
def test_nvfp4_output_is_identical_to_full_layer_copy(monkeypatch, tokens):
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    cache, _ = _cache(monkeypatch)
    torch.manual_seed(721)
    hidden = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 3], [2, 0]], device="cuda", dtype=torch.int32).repeat(
        (tokens + 1) // 2, 1
    )[:tokens].contiguous()
    weights = torch.rand(tokens, 2, device="cuda", dtype=torch.float32)
    weights /= weights.sum(dim=-1, keepdim=True)

    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    expected = fused_experts_nvfp4(
        hidden.clone(), *cache.wait_prefill_layer(0), weights, ids, 8, "silu", False,
    )
    cache.release_prefill_layer(0)
    torch.cuda.synchronize()
    for bank in cache.bank_caches.values():
        bank.view(torch.uint8).fill_(255)
    cache.begin_prefill(tokens)
    cache.prefetch_routed_prefill_layer(0, ids)
    actual = fused_experts_nvfp4(
        hidden.clone(), *cache.wait_prefill_layer(0), weights, ids, 8, "silu", False,
    )
    cache.release_prefill_layer(0)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_unavailable_batch_api_preserves_full_copy(monkeypatch):
    cache, sources = _cache(monkeypatch)
    monkeypatch.setattr(cache, "_resolve_batch_memcpy", lambda: False)
    cache.begin_prefill(31)
    assert not cache.prefill_selective_active
    cache.prefetch_prefill_layer(0)
    for view, source in zip(cache.wait_prefill_layer(0), sources.values()):
        assert torch.equal(view.view(torch.uint8).cpu(), source[0].view(torch.uint8))
    cache.release_prefill_layer(0)
