from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.moe.session_profile import (
    SESSION_EXPERT_PROFILE_TOPK,
    SessionExpertProfile,
    SessionProtectionRegistry,
    plan_session_prefetch,
    profile_storage_bytes,
    update_profile_sketch,
)


def _profile() -> SessionExpertProfile:
    return SessionExpertProfile(
        ids=((1, 2, 3), (4, 5), (6, 7), (8,)),
        counts=((9.0, 3.0, 1.0), (8.0, 2.0), (7.0, 4.0), (6.0,)),
    )


def test_profile_tensor_round_trip_and_versioned_absence():
    profile = _profile()
    restored = SessionExpertProfile.from_tensors(profile.to_tensors())
    assert restored is not None
    assert restored.ids == profile.ids
    assert restored.counts == profile.counts
    assert SessionExpertProfile.from_tensors({}) is None


def test_profile_storage_is_a_few_kibibytes_for_64_layers():
    assert profile_storage_bytes(64) == 2 + 64 * SESSION_EXPERT_PROFILE_TOPK * 4
    assert profile_storage_bytes(64) < 3 * 1024


def test_decode_capture_sketch_keeps_decayed_per_session_heavy_hitters():
    old_ids = torch.tensor([[1, 2, -1, -1], [3, -1, -1, -1]], dtype=torch.int32)
    old_counts = torch.tensor([[4.0, 2.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    routes = torch.tensor([[2, 2, 4], [5, 5, 5]], dtype=torch.int32)
    ids, counts = update_profile_sketch(old_ids, old_counts, routes, decay=0.5)
    rows = [
        dict(zip(row_ids, row_counts))
        for row_ids, row_counts in zip(ids.tolist(), counts.tolist())
    ]
    assert rows[0][2] == pytest.approx(3.0)
    assert rows[0][1] == pytest.approx(2.0)
    assert rows[0][4] == pytest.approx(1.0)
    assert rows[1][5] == pytest.approx(3.0)
    assert rows[1][3] == pytest.approx(2.5)


def test_admission_plan_splits_pinned_hot_and_cold_disk_rows():
    plan = plan_session_prefetch(
        _profile(),
        ["pinned", "disk", "locked", "disk"],
        hot_experts={1: (4,), 3: (99,)},
        protect_limit=4,
    )
    assert plan.promote == ((0, (1, 2, 3)), (1, (4,)))
    assert plan.willneed == ((1, (5,)), (3, (8,)))
    assert plan.protected == ((0, 1), (1, 4), (2, 6), (3, 8))
    assert plan.expert_count == 6


def test_protection_is_bounded_per_live_profile_and_empty_limit_disables_it():
    bounded = plan_session_prefetch(_profile(), ["pinned"] * 4, protect_limit=3)
    assert len(bounded.protected) == 3
    assert len(set(bounded.protected)) == 3
    assert plan_session_prefetch(_profile(), ["pinned"] * 4, protect_limit=0).protected == ()


def test_live_protection_union_is_bounded_and_releases_on_park():
    registry = SessionProtectionRegistry()
    first = plan_session_prefetch(_profile(), ["pinned"] * 4, protect_limit=2)
    second = plan_session_prefetch(_profile(), ["pinned"] * 4, protect_limit=3)
    registry.admit(10, first.protected)
    registry.admit(11, second.protected)
    assert len(registry.all()) <= 5
    assert registry.release(10) == first.protected
    assert registry.all() == frozenset(second.protected)


def test_session_prefetch_flag_validation_and_defaults():
    base = dict(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )
    default = EngineConfig(**base)
    assert default.session_expert_prefetch == "on"
    assert default.session_protect_experts == 64
    assert EngineConfig(**base, session_expert_prefetch="off").session_expert_prefetch == "off"
    with pytest.raises(ValueError, match="--session-expert-prefetch"):
        EngineConfig(**base, session_expert_prefetch="maybe")
    with pytest.raises(ValueError, match="--session-protect-experts"):
        EngineConfig(**base, session_protect_experts=-1)


def test_flag_off_gates_admission_lookup_before_touching_prefix_state():
    from freetoken.scheduler.cache import CacheManager

    manager = CacheManager.__new__(CacheManager)
    manager.is_hybrid = True
    manager.cache_type = "hybrid_radix"
    manager.moe_offload_cache = SimpleNamespace(session_profile_enabled=False)
    assert manager.lookup_expert_profile(torch.tensor([1, 2, 3])) is None


def test_prefetch_hook_fires_when_request_enters_waiting_queue():
    from freetoken.scheduler.prefill import PrefillManager

    calls = []
    profile = _profile()
    cache = SimpleNamespace(
        admit_expert_profile=lambda uid, ids: calls.append((uid, ids.tolist())) or profile
    )
    manager = PrefillManager(cache, SimpleNamespace(), SimpleNamespace())
    msg = SimpleNamespace(
        uid=77,
        input_ids=torch.tensor([4, 5, 6], dtype=torch.int32),
        sampling_params=SimpleNamespace(max_tokens=8),
        mm_embeds=None,
        priority=0,
        arrival_time=1.0,
    )

    manager.add_one_req(msg)

    assert calls == [(77, [4, 5, 6])]
    assert manager.pending_list[0].expert_profile is profile


def test_hybrid_vram_radix_hit_carries_parked_profile():
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

    radix = HybridRadixCache(torch.device("cpu"), page_size=1)
    token_ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    locations = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
    profile = _profile()
    radix.insert(token_ids, locations, mamba_value=3, expert_profile=profile)

    match = radix.match_prefix(token_ids)

    assert match.cached_len == 4
    assert match.node.expert_profile is profile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-gated advisory invariance")
def test_cuda_advisory_prefetch_does_not_change_moe_outputs():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(7)
    device = torch.device("cuda")
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=16,
    ).to(device)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=6,
        device=device,
    )
    cache.set_bank_sources({
        "gate_up": [torch.randn(4, 32, 8, pin_memory=True)],
        "down": [torch.randn(4, 8, 16, pin_memory=True)],
    })
    cache.configure_session_profiles(
        max_sessions=1, enabled=True, protect_experts=2, half_life_steps=100
    )
    layer.offload_cache = cache
    hidden = torch.randn(1, 8, device=device)
    router = torch.tensor([[4.0, 3.0, 2.0, 1.0]], device=device)

    expected = layer.decode_forward(hidden, router)
    cache.reset()
    cache.admit_session_profile(
        1, SessionExpertProfile(ids=((0, 1),), counts=((5.0, 4.0),))
    )
    actual = layer.decode_forward(hidden, router)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
