"""GPU-free tests for online HOT expert adaptation policy."""

from __future__ import annotations

import pytest

from freetoken.moe.hot_adapt import (
    HotSwap,
    finish_hot_swaps,
    plan_hot_swaps,
    recompute_hot_partition,
    retire_hot_swaps,
    update_decayed_counts,
)


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
