"""GPU-free tests for online HOT expert adaptation policy."""

from __future__ import annotations

import pytest

from freetoken.moe.hot_adapt import (
    HOT_STAGING_HEADROOM_BYTES,
    HotSwap,
    finish_hot_swaps,
    hot_staging_budget_bytes,
    hot_staging_rows,
    plan_hot_swaps,
    recompute_hot_partition,
    retire_hot_swaps,
    update_decayed_counts,
)


def test_staging_geometry_is_bounded_by_swap_delta_plus_headroom():
    expert_bytes = 13 << 20
    max_swap_bytes = 512 << 20
    rows = hot_staging_rows(max_swap_bytes, expert_bytes)

    assert rows * expert_bytes <= max_swap_bytes
    assert hot_staging_budget_bytes(max_swap_bytes) == (
        max_swap_bytes + HOT_STAGING_HEADROOM_BYTES
    )
    assert hot_staging_rows(expert_bytes - 1, expert_bytes) == 1


def test_decayed_counter_reaches_one_half_after_one_half_life():
    counts = update_decayed_counts(
        (0.0, 0.0), (1.0, 0.0), half_life_steps=4,
    )
    for _ in range(4):
        counts = update_decayed_counts(
            counts, (0.0, 0.0), half_life_steps=4,
        )
    assert counts == pytest.approx((0.5, 0.0))


def test_decayed_counter_accumulates_new_routes_after_decay():
    assert update_decayed_counts(
        (8.0, 4.0), (1.0, 3.0), half_life_steps=2, elapsed_steps=2,
    ) == pytest.approx((5.0, 5.0))


def test_partition_recompute_uses_same_equal_per_layer_byte_budget():
    counts = {0: (9.0, 1.0, 8.0), 2: (2.0, 7.0, 7.0)}
    assert recompute_hot_partition(
        counts,
        frozenset({0, 2}),
        budget_bytes=499,
        expert_bytes=100,
        num_experts=3,
    ) == {0: (0, 2), 2: (1, 2)}


def test_swap_planner_honors_global_byte_bound_and_prioritizes_gain():
    counts = {0: (1.0, 10.0, 9.0), 1: (1.0, 8.0, 7.0)}
    owners = {0: (0, None), 1: (0, None)}
    desired = {0: (1, 2), 1: (1, 2)}
    swaps = plan_hot_swaps(
        counts, owners, desired, expert_bytes=100, max_swap_bytes=200,
    )
    assert len(swaps) == 2
    assert len(swaps) * 100 <= 200
    assert {(swap.layer_id, swap.incoming_expert) for swap in swaps} == {(0, 1), (0, 2)}
    assert plan_hot_swaps(
        counts, owners, desired, expert_bytes=100, max_swap_bytes=99,
    ) == ()


def test_torn_mapping_guard_requires_copy_ack_before_publish():
    mapping = [[0, -1, -1]]
    swap = HotSwap(layer_id=0, row=0, incoming_expert=1, outgoing_expert=0)
    retired = retire_hot_swaps(mapping, (swap,))
    assert retired == [[-1, -1, -1]]

    with pytest.raises(RuntimeError, match="before copy"):
        finish_hot_swaps(retired, (swap,), copied_rows=set())

    assert finish_hot_swaps(
        retired, (swap,), copied_rows={(0, 0)}
    ) == [[-1, 0, -1]]


def test_synthetic_banks_retire_stage_copy_and_flip_without_host_mirror():
    import torch

    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = {
        "gate_up": [torch.arange(4 * 3, dtype=torch.int32).view(4, 3)],
        "down": [torch.arange(4 * 2, dtype=torch.int32).view(4, 2) + 100],
    }
    expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in sources.values())
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"),
        prefill_overlap=False, decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (0, 2)},
        hot_expert_capacity={0: 2},
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=1,
        max_swap_bytes=expert_bytes,
        expert_bytes=expert_bytes,
    )
    try:
        assert not getattr(cache, "hot_bank_sources", {})
        assert sum(t.numel() * t.element_size() for t in cache._hot_staging) <= expert_bytes
        assert getattr(cache, "hot_staging_bytes", 0) <= (
            expert_bytes + HOT_STAGING_HEADROOM_BYTES
        )
        old_slot = cache._hot_slot_for_row[0][0]
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][0])

        swap = HotSwap(0, 0, incoming_expert=1, outgoing_expert=0)
        cache._retire_hot_adaptation_swaps((swap,))
        cache._hot_adapt_future.result(timeout=5)
        cache._poll_hot_adaptation()

        assert cache.hot_row_for_expert[0].tolist() == [-1, 0, 1, -1]
        assert cache.slot_for_id[0, 0].item() == -1
        assert cache.slot_for_id[0, 1].item() == old_slot
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][1])
        assert torch.equal(cache.bank_caches["down"][old_slot], sources["down"][0][1])

        cache.reset()
        assert cache.slot_for_id[0, 1].item() == old_slot
        assert cache.usage[old_slot].item() == torch.iinfo(torch.int64).max
        assert torch.equal(cache.bank_caches["gate_up"][old_slot], sources["gate_up"][0][1])
    finally:
        cache.shutdown_hot_adaptation()
