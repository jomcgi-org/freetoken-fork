"""Pure planning and bookkeeping for online HOT expert adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


# Covers the fixed mapped-host mapping/snapshot tensors and one row of rounding
# when max_swap_bytes is smaller than, or not divisible by, an expert row.
HOT_STAGING_HEADROOM_BYTES = 64 << 20
HOT_ADAPT_TARGET_FILL_TOKENS = 2000
HOT_ADAPT_STEADY_INTERVAL_STEPS = 1000
HOT_ADAPT_MAX_STAGING_FRACTION = 0.25


@dataclass
class HotAdaptIntervalController:
    """Allocation-derived HOT adaptation cadence and its runtime state."""

    auto: bool
    fill_ticks: int
    fill_interval: int
    steady_interval: int
    current_interval: int
    target_fill_tokens: int
    fill_complete: bool = False

    @classmethod
    def create(
        cls,
        interval_steps: str | int,
        *,
        hot_budget_bytes: int,
        max_swap_bytes: int,
        target_fill_tokens: int = HOT_ADAPT_TARGET_FILL_TOKENS,
        steady_interval: int = HOT_ADAPT_STEADY_INTERVAL_STEPS,
    ) -> HotAdaptIntervalController:
        if hot_budget_bytes <= 0 or max_swap_bytes <= 0:
            raise ValueError("HOT adaptation interval geometry must be positive")
        if target_fill_tokens <= 0 or steady_interval <= 0:
            raise ValueError("HOT adaptation interval targets must be positive")
        fill_ticks = (hot_budget_bytes + max_swap_bytes - 1) // max_swap_bytes
        fill_interval = max(1, target_fill_tokens // fill_ticks)
        auto = interval_steps == "auto"
        if not auto and (
            isinstance(interval_steps, bool)
            or not isinstance(interval_steps, int)
            or interval_steps < 0
        ):
            raise ValueError("HOT adaptation interval must be 'auto' or non-negative")
        current = fill_interval if auto else int(interval_steps)
        return cls(
            auto=auto,
            fill_ticks=fill_ticks,
            fill_interval=fill_interval,
            steady_interval=steady_interval,
            current_interval=current,
            target_fill_tokens=target_fill_tokens,
        )

    def complete_tick(
        self,
        *,
        partition_full: bool,
        tick_interval: int,
        staging_seconds: float,
        covered_seconds: float,
    ) -> tuple[bool, bool, int]:
        """Apply a completed tick and return switch, back-off, and back-off floor."""
        if not self.auto:
            return False, False, self.current_interval

        backed_off = (
            covered_seconds > 0
            and staging_seconds
            > HOT_ADAPT_MAX_STAGING_FRACTION * covered_seconds
        )
        backoff_interval = max(self.fill_interval, tick_interval * 2)
        if backed_off:
            self.current_interval = max(self.current_interval, backoff_interval)

        switched = not self.fill_complete and partition_full
        if switched:
            self.fill_complete = True
            self.current_interval = max(
                self.current_interval,
                self.fill_interval,
                self.steady_interval,
            )
        return switched, backed_off, backoff_interval


@dataclass(frozen=True)
class HotSwap:
    """Install ``incoming_expert`` into one fixed HOT bank row."""

    layer_id: int
    row: int
    incoming_expert: int
    outgoing_expert: int | None


def hot_staging_rows(max_swap_bytes: int, expert_bytes: int) -> int:
    """Rows in the reusable host staging bank.

    Adaptation itself remains bounded by ``floor(max_swap / expert_bytes)``.
    One row is retained when that quotient is zero so a profiled initial set and
    a runtime cache rebuild can still be streamed without a full host mirror.
    """
    if max_swap_bytes < 0 or expert_bytes <= 0:
        raise ValueError("HOT staging requires a non-negative swap bound and positive rows")
    return max(1, max_swap_bytes // expert_bytes)


def hot_staging_budget_bytes(max_swap_bytes: int) -> int:
    """Conservative governor charge for staging payload plus fixed control data."""
    if max_swap_bytes < 0:
        raise ValueError("HOT staging swap bound must be non-negative")
    return max_swap_bytes + HOT_STAGING_HEADROOM_BYTES


def decay_multiplier(half_life_steps: int, elapsed_steps: int = 1) -> float:
    """Return the exponential multiplier for the requested number of steps."""
    if half_life_steps <= 0:
        raise ValueError("decay half-life must be positive")
    if elapsed_steps < 0:
        raise ValueError("elapsed decay steps must be non-negative")
    # math.exp2 is Python 3.11+; serving images still run 3.10
    return 2.0 ** (-float(elapsed_steps) / float(half_life_steps))


def update_decayed_counts(
    previous: Sequence[float],
    routed: Sequence[float],
    *,
    half_life_steps: int,
    elapsed_steps: int = 1,
) -> tuple[float, ...]:
    """CPU reference for one exact decay-and-add accumulator update."""
    if len(previous) != len(routed):
        raise ValueError("previous and routed counts must have the same length")
    factor = decay_multiplier(half_life_steps, elapsed_steps)
    return tuple(float(old) * factor + float(new) for old, new in zip(previous, routed))


def recompute_hot_partition(
    expert_counts: Mapping[int, Sequence[float]],
    disk_layer_ids: frozenset[int],
    *,
    budget_bytes: int,
    expert_bytes: int,
    num_experts: int,
) -> dict[int, tuple[int, ...]]:
    """Select an equal top-N partition under the configured resident-byte budget."""
    if budget_bytes < 0 or expert_bytes <= 0 or num_experts <= 0:
        raise ValueError("HOT planner geometry must be non-negative with positive rows")
    if not disk_layer_ids or budget_bytes == 0:
        return {}
    missing = set(disk_layer_ids) - set(expert_counts)
    if missing:
        raise ValueError(f"counts have no entries for DISK layers {sorted(missing)}")
    top_n = min(
        num_experts,
        budget_bytes // (expert_bytes * len(disk_layer_ids)),
    )
    if top_n <= 0:
        return {}
    result = {}
    for layer_id in sorted(disk_layer_ids):
        counts = expert_counts[layer_id]
        if len(counts) != num_experts:
            raise ValueError(
                f"counts layer {layer_id} has {len(counts)} experts, expected {num_experts}"
            )
        ranked = sorted(
            range(num_experts),
            key=lambda expert_id: (-float(counts[expert_id]), expert_id),
        )
        result[layer_id] = tuple(sorted(ranked[:top_n]))
    return result


def plan_hot_swaps(
    expert_counts: Mapping[int, Sequence[float]],
    slot_owners: Mapping[int, Sequence[int | None]],
    desired: Mapping[int, Sequence[int]],
    *,
    expert_bytes: int,
    max_swap_bytes: int,
) -> tuple[HotSwap, ...]:
    """Plan deterministic highest-gain row replacements within one byte bound."""
    if expert_bytes <= 0 or max_swap_bytes < 0:
        raise ValueError("swap planner requires positive expert bytes and a non-negative bound")
    max_swaps = max_swap_bytes // expert_bytes
    if max_swaps <= 0:
        return ()

    candidates: list[tuple[float, int, int, HotSwap]] = []
    for layer_id in sorted(desired):
        counts = expert_counts[layer_id]
        owners = tuple(slot_owners[layer_id])
        current = {owner for owner in owners if owner is not None}
        target = set(int(expert) for expert in desired[layer_id])
        incoming = sorted(target - current, key=lambda expert: (-counts[expert], expert))
        free_rows = [row for row, owner in enumerate(owners) if owner is None]
        outgoing_rows = sorted(
            (
                (float(counts[owner]), owner, row)
                for row, owner in enumerate(owners)
                if owner is not None and owner not in target
            ),
            key=lambda item: (item[0], -item[1], item[2]),
        )
        available = [(0.0, None, row) for row in free_rows] + outgoing_rows
        for incoming_expert, (out_score, outgoing_expert, row) in zip(incoming, available):
            gain = float(counts[incoming_expert]) - out_score
            swap = HotSwap(layer_id, row, incoming_expert, outgoing_expert)
            candidates.append((-gain, layer_id, incoming_expert, swap))

    candidates.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in candidates[:max_swaps])


def retire_hot_swaps(
    mapping: Sequence[Sequence[int]], swaps: Sequence[HotSwap],
) -> list[list[int]]:
    """Remove outgoing mappings before their rows can be used as staging."""
    retired = [list(layer) for layer in mapping]
    for swap in swaps:
        layer = retired[swap.layer_id]
        if layer[swap.incoming_expert] >= 0:
            raise RuntimeError("incoming HOT expert is already mapped")
        if swap.outgoing_expert is not None:
            if layer[swap.outgoing_expert] != swap.row:
                raise RuntimeError("outgoing HOT row ownership changed before retirement")
            layer[swap.outgoing_expert] = -1
    return retired


def finish_hot_swaps(
    mapping: Sequence[Sequence[int]],
    swaps: Sequence[HotSwap],
    copied_rows: set[tuple[int, int]],
) -> list[list[int]]:
    """Publish only rows whose complete bank copies have been acknowledged."""
    finished = [list(layer) for layer in mapping]
    for swap in swaps:
        key = (swap.layer_id, swap.row)
        if key not in copied_rows:
            raise RuntimeError(
                f"refusing to publish HOT layer {swap.layer_id} row {swap.row} before copy"
            )
        layer = finished[swap.layer_id]
        if layer[swap.incoming_expert] >= 0:
            raise RuntimeError("incoming HOT expert became mapped before copy completion")
        layer[swap.incoming_expert] = swap.row
    return finished
