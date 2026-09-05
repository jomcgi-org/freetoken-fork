"""Protected-slot routing oracle reporting under --moe-collect-stats."""

import contextlib
import io
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.scheduler.scheduler import _moe_oracle_status_fragment
from freetoken.server.args import ServerArgs, parse_args


def test_flag_is_registered_and_defaults_off():
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "--moe-collect-stats" in output.getvalue()
    assert ServerArgs.moe_collect_stats is False


def _cache(*, collect_stats: bool) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.cpu_executor = SimpleNamespace(
        disk_prefetch_stats=lambda reset=False: {},
        reset_disk_lookahead=lambda: None,
    )
    cache.hot_expert_capacity = {0: 1, 1: 2}
    cache.collect_stats = collect_stats
    return cache


def test_protected_oracle_uses_routed_pairs_and_resets_its_window():
    cache = _cache(collect_stats=True)
    cache.decode_freq.copy_(
        torch.tensor([[5, 3, 2, 0], [4, 4, 1, 1]], dtype=torch.int64)
    )
    cache.stat_hot_pairs.fill_(9)
    cache.stat_hot_total_pairs.fill_(20)

    stats = cache.disk_prefetch_stats(reset=True)

    # Same-capacity oracle: top 1 from layer 0 plus top 2 from layer 1.
    assert stats["oracle_hits"] == 13
    assert stats["oracle_routed_pairs"] == 20
    assert stats["oracle_hit"] == pytest.approx(13 / 20)
    assert stats["realized_hit"] == pytest.approx(9 / 20)

    cache.decode_freq.add_(
        torch.tensor([[0, 2, 0, 0], [0, 0, 3, 1]], dtype=torch.int64)
    )
    cache.stat_hot_pairs.fill_(2)
    cache.stat_hot_total_pairs.fill_(6)
    next_stats = cache.disk_prefetch_stats(reset=True)
    assert next_stats["oracle_hits"] == 6
    assert next_stats["oracle_hit"] == pytest.approx(1.0)
    assert next_stats["realized_hit"] == pytest.approx(2 / 6)


def test_protected_oracle_is_gated_by_moe_collect_stats():
    cache = _cache(collect_stats=False)
    cache.stat_hot_pairs.fill_(3)
    cache.stat_hot_total_pairs.fill_(4)
    stats = cache.disk_prefetch_stats(reset=True)
    assert "oracle_hit" not in stats
    assert "realized_hit" not in stats


def test_rebuild_clears_the_protected_oracle_baseline():
    cache = _cache(collect_stats=True)
    cache.set_bank_sources(
        {
            "gate_up": [torch.randn(4, 8, 4) for _ in range(2)],
            "down": [torch.randn(4, 4, 8) for _ in range(2)],
        }
    )
    cache.hot_expert_capacity = {0: 1, 1: 2}
    cache.decode_freq.fill_(10)
    cache.protected_routing_stats(realized_hits=0, routed_pairs=80, reset=True)

    cache.rebuild(8)
    cache.decode_freq.copy_(
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int64)
    )
    stats = cache.protected_routing_stats(
        realized_hits=2, routed_pairs=2, reset=True
    )

    assert stats["oracle_hits"] >= 2
    assert stats["oracle_hit"] >= stats["realized_hit"]


def test_stats_line_reports_oracle_against_realized_coverage():
    fragment = _moe_oracle_status_fragment(
        {"oracle_hit": 0.8125, "realized_hit": 0.6875}
    )
    assert fragment == ", disk oracle_hit: 81.25% vs realized: 68.75%"
    assert _moe_oracle_status_fragment({"hot_pair_rate": 0.5}) == ""


@pytest.mark.parametrize("collect,timing", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("all_hot", [False, True])
def test_decode_classification_only_runs_with_diagnostics(
    monkeypatch, collect, timing, all_hot,
):
    from freetoken.layers import moe

    hidden = torch.ones(1, 8)
    classified = []
    recorded = []
    cpu_work = []
    original_classify = moe._all_hot

    def classify(ids):
        assert collect or timing, "disabled telemetry launched a route reduction"
        result = original_classify(ids)
        classified.append(bool(result))
        return result

    monkeypatch.setattr(moe, "_all_hot", classify)
    executor = SimpleNamespace(
        _step_timing=timing,
        record_all_hot=lambda value: recorded.append(bool(value)),
        decode_submit=lambda *args: cpu_work.append(args[3].clone()),
        decode_sync=lambda *args, **kwargs: hidden * 2,
    )
    cache = SimpleNamespace(
        collect_stats=collect, cpu_executor=executor,
        copy_missing=lambda: None, bank_views=lambda: (),
        alphas_for_slots=lambda layer_id: None,
    )
    layer = SimpleNamespace(
        layer_id=0, _expert_gemm=lambda *args, **kwargs: hidden * 3,
    )
    raw = torch.tensor([[0, 1]], dtype=torch.int32)
    slots = torch.tensor([[3, 4 if all_hot else -1]], dtype=torch.int32)
    result = moe.OffloadMoELayer._decode_split_partials(
        layer, cache, hidden, torch.ones(1, 2), slots, raw, count_all_hot=True,
    )

    assert torch.equal(result, hidden * 5)
    assert classified == ([all_hot] if collect or timing else [])
    assert recorded == classified
    assert len(cpu_work) == 1
    assert cpu_work[0].tolist() == [[-1, -1 if all_hot else 1]]


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("executed_swaps", [0, 1])
def test_idle_completion_gates_live_history_readback(
    monkeypatch, collect, executed_swaps,
):
    from freetoken.moe import offload_cache
    from freetoken.moe.hot_adapt import HotAdaptIntervalController

    cache = _cache(collect_stats=collect)
    cache._hot_adapt_interval_controller = HotAdaptIntervalController.create(
        64, hot_budget_bytes=256, max_swap_bytes=64,
    )
    cache._hot_adapt_tick_interval_tokens = 64
    cache._hot_adapt_tick_covered_seconds = 1.0
    cache._hot_adapt_tick_boundary = "idle"
    cache._hot_adapt_tick_planned_swaps = 1
    cache._hot_adapt_tick_executed_swaps = executed_swaps
    cache._hot_adapt_tick_rate_before = 0.25
    events = []
    completed = []
    messages = []
    monkeypatch.setattr(
        cache, "_checkpoint_published_hot_slot_owners",
        lambda: events.append("checkpoint"),
    )
    monkeypatch.setattr(cache, "snapshot_hot_plan", lambda: events.append("snapshot"))

    def read_live_histories():
        assert collect, "disabled diagnostics read live GPU histories"
        events.append("readback")
        return 0.75

    monkeypatch.setattr(cache, "decayed_hot_pair_rate", read_live_histories)
    monkeypatch.setattr(offload_cache.logger, "info_rank0", messages.append)
    cache._hot_adapt_idle_tracker = SimpleNamespace(
        tick_completed=lambda now, swaps: completed.append((now, swaps)),
    )

    cache._complete_hot_adaptation_tick(staging_seconds=0.0)

    assert events == (
        ["checkpoint"] + (["snapshot"] if executed_swaps else [])
        + (["readback"] if collect else [])
    )
    assert len(completed) == 1 and completed[0][1] == executed_swaps
    assert len(messages) == 1
    assert f"executed_swaps={executed_swaps}" in messages[0]
    assert ("decayed_hot_pair_rate=25.00%->75.00%" in messages[0]) is collect
