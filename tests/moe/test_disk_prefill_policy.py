"""Staged prefill selection, CLI validation, and scratch-slot reservations."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args


@pytest.mark.parametrize("threshold", [None, 1536])
def test_staged_cli_crossover_default_and_override(threshold):
    args, _ = parse_args([
        "--model", "/tmp/nonexistent-model", "--dtype", "bfloat16",
        "--moe-disk-prefill", "staged",
    ] + ([] if threshold is None else ["--moe-disk-prefill-min-tokens", str(threshold)]))
    assert args.moe_disk_prefill == "staged"
    assert args.moe_disk_prefill_min_tokens == (1024 if threshold is None else threshold)
    assert EngineConfig.__dataclass_fields__["moe_disk_prefill_min_tokens"].default == 1024
    assert OffloadMoeCache.__dataclass_fields__["moe_disk_prefill_min_tokens"].default == 1024


@pytest.mark.parametrize("kwargs,message", [
    ({"moe_disk_prefill_min_tokens": 0}, "must be positive"),
    ({"moe_disk_prefill_min_tokens": -1}, "must be positive"),
    ({"moe_disk_decode": "gpufetch"}, "requires --moe-disk-decode cpu"),
])
def test_staged_config_rejects_invalid_policy(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EngineConfig(
            model_path="/tmp/model", tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16, moe_disk_prefill="staged", **kwargs,
        )


@pytest.mark.parametrize("overlap", [False, True])
def test_chunk_threshold_rebuilds_pinned_schedule_across_cpu_fallback(overlap):
    cache = OffloadMoeCache(
        num_layers=4, num_experts=8, cache_size=24, device=torch.device("cpu"),
        prefill_overlap=overlap, moe_disk_prefill="staged",
        moe_disk_prefill_min_tokens=512,
    )
    cache.layer_residency = ["disk", "pinned", "disk", "pinned"]
    cache._unpinned_layers = frozenset({0, 2})
    cache._configure_prefill_overlap_layers()
    for tokens, mode in [(511, "cpu"), (512, "staged"), (4096, "staged"), (12, "cpu")]:
        cache.begin_prefill(tokens)
        assert cache.effective_disk_prefill == mode
        expected = [-1, 1, -1, 1] if mode == "staged" else [-1, 0, -1, 1]
        assert cache._prefill_overlap_buffer_ids == (expected if overlap else [-1] * 4)


@pytest.mark.parametrize("overlap", [False, True])
def test_staging_rebuild_reserves_scratch_before_protected_rows(overlap):
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=24, device=torch.device("cpu"),
        prefill_overlap=overlap, moe_disk_prefill="staged",
    )
    cache.hot_expert_capacity = {0: 2, 1: 2}
    cache.validate_rebuild(24)
    cache.validate_rebuild(12)  # Below 2E, rebuild disables overlap.
    with pytest.raises(ValueError, match="unprotected GPU slots"):
        cache.validate_rebuild(8)
    if overlap:
        with pytest.raises(ValueError, match="unprotected GPU slots"):
            cache.validate_rebuild(16)


@pytest.mark.parametrize("disk,uffd,path", [
    (False, False, "/weights.ftw"), (True, True, "/weights.ftw"), (True, False, None),
])
def test_staging_rejects_unsupported_bank_before_allocating_cuda_buffers(disk, uffd, path):
    cache = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=24, device=torch.device("cpu"),
        quant_format="nvfp4", moe_disk_prefill="staged",
    )
    # No CUDA allocation is needed to validate source ownership.
    cache.device = torch.device("cuda")
    cache.layer_residency = ["disk"]
    source = SimpleNamespace(_freetoken_host_bank=SimpleNamespace(
        _disk=disk, _uffd=uffd, _file_path=path,
    ))
    cache.banks = [([source], None)]
    with pytest.raises(ValueError, match="ordinary file-backed banks"):
        cache.init_disk_prefill_staging()
    assert cache._disk_prefill_staging is None
