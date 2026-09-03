from __future__ import annotations

import torch


class _CPUByteStore:
    def __init__(self, element_size: int, calls: list[tuple]) -> None:
        self.element_size = element_size
        self.calls = calls

    def launch(self, k_cache, v_cache, indices, k, v) -> None:
        assert k_cache.dtype == v_cache.dtype == k.dtype == v.dtype == torch.uint8
        assert k_cache.shape[1] == v_cache.shape[1] == self.element_size
        assert k.shape[1] == v.shape[1] == self.element_size
        self.calls.append((k_cache, v_cache, indices, k, v))
        for source_row, destination_row in enumerate(indices.tolist()):
            k_cache[destination_row].copy_(k[source_row])
            v_cache[destination_row].copy_(v[source_row])


def _patch_cpu_byte_store(monkeypatch) -> list[tuple]:
    from freetoken.kernel import store

    calls: list[tuple] = []
    monkeypatch.setattr(
        store,
        "_jit_store_module",
        lambda element_size: _CPUByteStore(element_size, calls),
    )
    return calls


def test_store_cache_accepts_int32_indices(monkeypatch):
    from freetoken.kernel.store import store_cache

    calls = _patch_cpu_byte_store(monkeypatch)
    k_cache = torch.zeros(5, 2, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    k = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    v = -k
    indices = torch.tensor([3, 1], dtype=torch.int32)

    store_cache(k_cache, v_cache, indices, k, v)

    assert calls[0][2].dtype == torch.int32
    assert torch.equal(k_cache[indices.to(torch.long)], k)
    assert torch.equal(v_cache[indices.to(torch.long)], v)


def test_store_cache_converts_e4m3_without_preclamp_then_copies_bytes(monkeypatch):
    from freetoken.kernel.store import store_cache

    calls = _patch_cpu_byte_store(monkeypatch)
    k_cache = torch.zeros(4, 3, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    k = torch.tensor([[500.0, -500.0, float("nan")], [0.5, -2.0, 17.0]])
    v = -k
    indices = torch.tensor([2, 0], dtype=torch.int32)

    def reject_clamp(*_args, **_kwargs):
        raise AssertionError("store_cache must not make a separate clamp pass")

    monkeypatch.setattr(torch.Tensor, "clamp", reject_clamp)
    store_cache(k_cache, v_cache, indices, k, v)

    expected_k = k.to(torch.float8_e4m3fn).view(torch.uint8)
    expected_v = v.to(torch.float8_e4m3fn).view(torch.uint8)
    assert calls[0][0].dtype == calls[0][3].dtype == torch.uint8
    assert torch.equal(k_cache[indices.to(torch.long)].view(torch.uint8), expected_k)
    assert torch.equal(v_cache[indices.to(torch.long)].view(torch.uint8), expected_v)


def test_aot_store_prebuilds_bf16_and_fp8_widths_for_every_model():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS, store_element_sizes

    for model in SUPPORTED_MODELS:
        expected = {
            kv_heads * head_dim * dtype_bytes
            for kv_heads, head_dim in model.kv_groups
            for dtype_bytes in (1, 2)
        }
        assert store_element_sizes(model) == expected, model.name
