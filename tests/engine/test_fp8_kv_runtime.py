from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "kv_dtype, expected_scale_keys",
    [
        (torch.bfloat16, set()),
        (torch.float8_e4m3fn, {"k_scale", "v_scale"}),
    ],
)
def test_flashinfer_run_scales_are_fp8_only(kv_dtype, expected_scale_keys):
    from freetoken.attention.fi import FIMetadata, FlashInferBackend

    seen = {}

    class Wrapper:
        def run(self, **kwargs):
            seen.update(kwargs)
            return kwargs["q"]

    cache = torch.zeros(2, 1, 1, 2, dtype=kv_dtype)
    kv_cache = SimpleNamespace(
        store_kv=lambda *args: None,
        k_cache=lambda layer: cache,
        v_cache=lambda layer: cache,
        k_scale=1.0,
        v_scale=1.0,
    )
    metadata = object.__new__(FIMetadata)
    metadata.kv_dtype = kv_dtype
    metadata.wrapper = Wrapper()
    backend = object.__new__(FlashInferBackend)
    backend.kvcache = kv_cache
    backend._initialize_metadata_once = lambda value: None
    q = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=torch.tensor([0, 1]))

    assert backend.forward(q, q, q, 0, batch) is q
    assert ({"k_scale", "v_scale"} & seen.keys()) == expected_scale_keys


def test_fp8_pool_contract_requires_storage_compute_and_scales():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache import _validate_fp8_kv_pool
    from freetoken.kvcache.qsa_pool import QSAKVCache

    config = SimpleNamespace(kv_cache_dtype="fp8_e4m3", attention_backend="future_fp8")
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    valid = QSAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=64,
        num_pages=2,
        page_size=4,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device=torch.device("meta"),
        index_head_dim=32,
        num_index_layers=2,
        index_ratio=2,
        num_req_slots=2,
    )
    assert _validate_fp8_kv_pool(config, valid, torch.bfloat16) is valid

    incomplete = SimpleNamespace(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="has no matching FP8 pool"):
        _validate_fp8_kv_pool(config, incomplete, torch.bfloat16)


def test_cost_uses_pool_family_dtype_instead_of_fp8_flag():
    from freetoken.attention import AttnType
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    spec = SimpleNamespace(
        attn_type=AttnType.BSA,
        mla=False,
        head_dim=64,
        num_kv_heads=2,
        num_layers=2,
        index_head_dim=32,
        num_index_layers=2,
        index_ratio=1,
    )
    model = SimpleNamespace(kv_cache_group_specs=lambda: (spec,))
    config = SimpleNamespace(
        model_config=model,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_e4m3",
        tp_info=SimpleNamespace(size=1),
    )

    assert spec_kv_bytes_per_token(spec, config) == 2 * 64 * 2 * 2 * 2 + 32 * 2 * 2
