"""GPU-free tests for the reasoning/answer decode histories (--moe-hot-adapt-histories split3)."""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.hot_adapt import (
    HOT_PLAN_FILENAME,
    PHASE_TIEBREAK_BLEND,
    aim_histories,
    atomic_write_hot_plan,
    load_hot_plan,
    make_hot_plan_document,
)
from freetoken.scheduler.status import _hot_adapt_history_status_fragment

from .test_hot_adapt import (
    _offload_cache_class_without_triton,
    _offload_kernels_without_triton,
)
from .test_hot_plan_persistence import IDENTITY

ANSWER = {0: (8.0, 0.0, 2.0, 0.0)}
REASONING = {0: (0.0, 6.0, 0.0, 2.0)}
PREFILL = {0: (0.0, 20.0, 0.0, 4.0)}


def _aim(boundary, reasoning_phase, aim="phase", prefill=PREFILL, reasoning=REASONING):
    return aim_histories(
        ANSWER, prefill, reasoning,
        aim=aim, boundary=boundary, reasoning_phase=reasoning_phase, prefill_blend=0.25,
    )[0]


def test_phase_aim_decode_follows_reasoning_flag_with_tiebreak():
    t = PHASE_TIEBREAK_BLEND
    assert _aim("decode", True) == pytest.approx(
        tuple(r + t * a for r, a in zip(REASONING[0], ANSWER[0]))
    )
    assert _aim("decode", False) == pytest.approx(
        tuple(a + t * r for a, r in zip(ANSWER[0], REASONING[0]))
    )
    # Prefill never leaks into a decode aim, and idle ticks aim like decode.
    assert _aim("idle", True) == _aim("decode", True)


def test_phase_aim_prefill_boundary_aims_at_the_phase_about_to_run():
    t = PHASE_TIEBREAK_BLEND
    for reasoning_phase, aimed, other in ((True, REASONING, ANSWER), (False, ANSWER, REASONING)):
        expected = tuple(
            p + 0.25 * (a + t * o)
            for p, a, o in zip(PREFILL[0], aimed[0], other[0])
        )
        assert _aim("prefill", reasoning_phase) == pytest.approx(expected)


def test_blend_aim_sums_both_decode_histories_plus_prefill_share():
    expected = tuple(
        a + r + 0.25 * p for a, r, p in zip(ANSWER[0], REASONING[0], PREFILL[0])
    )
    assert _aim("decode", True, aim="blend") == pytest.approx(expected)
    assert _aim("decode", False, aim="blend") == pytest.approx(expected)


def test_split3_without_prefill_history_still_aims_by_phase():
    out = _aim("prefill", True, prefill=None)
    assert out == _aim("decode", True, prefill=None)


def test_split_and_shared_aim_are_unchanged():
    kwargs = dict(boundary="decode", reasoning_phase=True, prefill_blend=0.25)
    assert aim_histories(ANSWER, None, None, aim="phase", **kwargs) == ANSWER
    assert aim_histories(ANSWER, PREFILL, None, aim="phase", **kwargs) == ANSWER
    assert aim_histories(
        ANSWER, PREFILL, None, aim="phase", **{**kwargs, "boundary": "prefill"}
    )[0] == (2.0 - 2.0 + 0.0 + 0.25 * 8.0, 20.0, 0.5, 4.0)
    assert aim_histories(ANSWER, PREFILL, None, aim="blend", **kwargs)[0] == (
        8.0, 5.0, 2.0, 1.0,
    )


def test_hot_kernel_routes_decode_counts_by_device_phase_index(monkeypatch):
    from types import SimpleNamespace

    kernels = _offload_kernels_without_triton(monkeypatch)
    stack = torch.zeros((2, 1, 4))
    cache = SimpleNamespace(
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        hot_row_for_expert=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        slot_for_id=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        id_of_slot=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        usage=torch.zeros(4, dtype=torch.int64),
        step=torch.zeros((), dtype=torch.int64),
        active_mask=torch.zeros(4, dtype=torch.int32),
        evict_slots=torch.empty(4, dtype=torch.int32),
        src_indices=torch.empty(4, dtype=torch.int32),
        num_indices=torch.zeros(1, dtype=torch.int64),
        num_missing_full=torch.zeros(1, dtype=torch.int64),
        stat_hot_pairs=torch.zeros((), dtype=torch.int64),
        stat_hot_total_pairs=torch.zeros((), dtype=torch.int64),
        hot_adapt_enabled=True,
        _hot_decay_factor=1.0,
        decayed_decode_freq=stack[0],
        decayed_reasoning_freq=stack[1],
        decayed_prefill_freq=torch.zeros((1, 4)),
        decode_phase_idx=torch.zeros(1, dtype=torch.int32),
    )
    ids = lambda: torch.tensor([[0, 1]], dtype=torch.int32)  # noqa: E731

    kernels.ensure_experts_hot(cache, 0, ids())
    cache.decode_phase_idx.fill_(1)
    kernels.ensure_experts_hot(cache, 0, ids())
    kernels.ensure_experts_hot(cache, 0, ids())
    kernels.ensure_experts_hot(cache, 0, ids(), history="prefill", route_weight=0.5)

    assert stack[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]  # answer: one step
    assert stack[1, 0].tolist() == [2.0, 2.0, 0.0, 0.0]  # reasoning: two steps
    assert cache.decayed_prefill_freq[0].tolist() == [0.5, 0.5, 0.0, 0.0]


def _split3_cache(monkeypatch, histories="split3"):
    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    stack = torch.zeros((2, 1, 4))
    cache.decode_phase_idx = torch.zeros(1, dtype=torch.int32)
    cache._decode_phase_reasoning = False
    cache.decayed_decode_freq = stack[0]
    cache.decayed_reasoning_freq = stack[1] if histories == "split3" else None
    cache.hot_adapt_histories = histories
    return cache


def test_set_decode_phase_flips_device_index_only_on_change(monkeypatch):
    cache = _split3_cache(monkeypatch)
    fills = []
    real_fill = cache.decode_phase_idx.fill_
    cache.decode_phase_idx = type(
        "Idx", (), {"fill_": lambda self, v: (fills.append(v), real_fill(v))[1]}
    )()

    cache.set_decode_phase(False)
    cache.set_decode_phase(True)
    cache.set_decode_phase(True)
    cache.set_decode_phase(False)

    assert fills == [1, 0]
    assert cache._decode_phase_reasoning is False


def test_set_decode_phase_is_noop_without_reasoning_history(monkeypatch):
    cache = _split3_cache(monkeypatch, histories="split")
    cache.set_decode_phase(True)
    assert cache.decode_phase_idx.item() == 0
    assert cache._decode_phase_reasoning is False


def test_plan_uses_reasoning_history_while_reasoning(monkeypatch):
    from freetoken.moe import hot_adapt

    captured = []
    monkeypatch.setattr(
        hot_adapt, "recompute_hot_partition",
        lambda counts, *_a, **_k: (captured.append(counts), {0: (0,)})[1],
    )
    for reasoning, boundary in ((True, "decode"), (False, "decode"), (True, "prefill")):
        cache = _split3_cache(monkeypatch)
        cache._hot_adapt_snapshot_host = torch.tensor([ANSWER[0]])
        cache._hot_adapt_prefill_snapshot_host = torch.tensor([PREFILL[0]])
        cache._hot_adapt_reasoning_snapshot_host = torch.tensor([REASONING[0]])
        cache._hot_adapt_tick_reasoning = reasoning
        cache.hot_expert_capacity = {0: 1}
        cache.hot_adapt_aim = "phase"
        cache.hot_adapt_prefill_blend = 0.25
        cache.hot_adapt_expert_bytes = 1
        cache.num_experts = 4
        cache._hot_slot_owners = {0: [0]}
        cache.hot_adapt_max_swap_bytes = 1
        cache.hot_adapt_hot_budget_bytes = 1
        cache.hot_adapt_boundary_cap_frac = 1.0
        cache._plan_hot_adaptation(
            None, token=1, swap_budget_bytes=1, boundary=boundary, tick_count=1
        )
    expected = [
        _aim("decode", True), _aim("decode", False), _aim("prefill", True),
    ]
    for got, want in zip(captured, expected):
        assert got[0] == pytest.approx(want)


def test_reset_clears_reasoning_history_and_reseeds(monkeypatch):
    import sys
    from types import ModuleType

    cache = _split3_cache(monkeypatch)
    kernels = ModuleType("freetoken.moe.offload_kernels")
    kernels.reset_cache = lambda _cache: None
    monkeypatch.setitem(sys.modules, "freetoken.moe.offload_kernels", kernels)
    cache.device = torch.device("cpu")
    cache.decayed_decode_freq.fill_(9.0)
    cache.decayed_reasoning_freq.fill_(7.0)
    cache.decayed_prefill_freq = None
    cache._hot_plan_counter_seed = {}
    cache._hot_plan_reasoning_counter_seed = {0: (1.0, 2.0, 3.0, 4.0)}
    cache._restore_hot_slot_metadata = lambda: None
    cache.expert_recency = torch.zeros((1, 4), dtype=torch.int64)
    cache.session_profile_ids = None
    cache.cpu_executor = None

    cache.reset()

    assert cache.decayed_decode_freq.sum().item() == 0.0
    assert cache.decayed_reasoning_freq[0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_status_reports_aimed_history_and_per_history_ticks(monkeypatch):
    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=4, device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.cpu_executor = type(
        "E", (), {"_disk_banks": {0: [object()]}, "disk_prefetch_stats": lambda s, reset=False: {"prefetch_calls": 0}}
    )()
    cache.hot_adapt_histories = "split3"
    cache.hot_adapt_aim = "phase"
    cache.decayed_prefill_freq = torch.zeros((1, 4))
    cache.decayed_reasoning_freq = cache._decayed_decode_stack[1]
    cache.decayed_decode_freq[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cache.decayed_reasoning_freq[0] = torch.tensor([0.0, 3.0, 0.0, 0.0])
    cache._decode_phase_reasoning = True
    cache.hot_adapt_ticks_reasoning = 5
    cache.hot_adapt_ticks_answer = 2

    stats = cache.disk_prefetch_stats(reset=True)

    assert stats["hot_adapt_histories"] == "split3"
    assert stats["decayed_reasoning_share"] == pytest.approx(0.75)
    assert stats["hot_adapt_decode_aim"] == "reasoning"
    assert stats["hot_adapt_ticks_reasoning"] == 5
    assert stats["hot_adapt_ticks_answer"] == 2
    assert "hot_adapt_decode_aim: reasoning" in _hot_adapt_history_status_fragment(stats)
    # The window resets: the next report only counts new ticks.
    cache.hot_adapt_ticks_answer += 1
    again = cache.disk_prefetch_stats(reset=True)
    assert (again["hot_adapt_ticks_reasoning"], again["hot_adapt_ticks_answer"]) == (0, 1)


def test_status_fragment_unchanged_for_split():
    assert _hot_adapt_history_status_fragment(
        {"hot_adapt_histories": "split", "decayed_prefill_share": 0.5}
    ) == "hot_adapt_histories: split, decayed_prefill_share: 50.00%"


# --- persistence -----------------------------------------------------------------


def _document(reasoning=None, prefill=None):
    return make_hot_plan_document(
        identity=IDENTITY,
        disk_layer_ids=(0, 1),
        num_layers=2,
        num_experts=4,
        hot_budget_bytes=400,
        tier_commit="tier-old",
        protected_slots={0: (0, 3), 1: (2, 1)},
        decayed_counters={0: (1.0, 9.0, 8.0, 2.0), 1: (7.0, 1.0, 6.0, 3.0)},
        decayed_prefill_counters=prefill,
        decayed_reasoning_counters=reasoning,
        written_at=1000.0,
    )


def _load(path):
    return load_hot_plan(
        str(path),
        identity=IDENTITY,
        disk_layer_ids=frozenset({0, 1}),
        num_layers=2,
        num_experts=4,
        current_capacity={0: 2, 1: 2},
        current_hot_budget_bytes=400,
        static_expert_ids={0: (1, 2), 1: (0, 3)},
        tier_commit="tier-new",
        current_capacity_policy="equal",
        now=1060.0,
    )


def test_reasoning_counters_round_trip(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    reasoning = {0: (5.0, 0.0, 1.0, 2.0), 1: (0.0, 4.0, 0.0, 1.0)}
    prefill = {0: (1.0, 1.0, 1.0, 1.0), 1: (2.0, 2.0, 2.0, 2.0)}
    document = _document(reasoning=reasoning, prefill=prefill)
    assert "decayed_reasoning_counters" in document
    atomic_write_hot_plan(str(path), document)

    seed = _load(path)

    assert seed.reasoning_counters[0] == pytest.approx(reasoning[0])
    assert seed.reasoning_counters[1] == pytest.approx(reasoning[1])
    assert seed.prefill_counters[0] == pytest.approx(prefill[0])
    assert seed.counters[0] == pytest.approx((1.0, 9.0, 8.0, 2.0))


def test_plan_without_reasoning_section_degrades_to_empty_reasoning_seed(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert "decayed_reasoning_counters" not in document
    atomic_write_hot_plan(str(path), document)

    assert _load(path).reasoning_counters == {}


def test_reasoning_counter_validation(tmp_path):
    with pytest.raises(ValueError, match="reasoning counter layer 0"):
        _document(reasoning={0: (1.0, 2.0), 1: (0.0, 0.0, 0.0, 1.0)})
    with pytest.raises(ValueError, match="layers must match"):
        _document(reasoning={0: (1.0, 2.0, 3.0, 4.0)})
    with pytest.raises(ValueError, match="invalid value"):
        _document(reasoning={0: (-1.0, 0.0, 0.0, 0.0), 1: (0.0,) * 4})


def test_split3_only_plan_with_zero_decode_counters_still_persists():
    document = make_hot_plan_document(
        identity=IDENTITY, disk_layer_ids=(0,), num_layers=1, num_experts=4,
        hot_budget_bytes=100, tier_commit="t", protected_slots={0: (0, 1)},
        decayed_counters={0: (0.0,) * 4},
        decayed_reasoning_counters={0: (3.0, 1.0, 0.0, 0.0)},
    )
    assert document is not None


def test_reasoning_seed_validated_and_applied(monkeypatch):
    cache = _split3_cache(monkeypatch)
    cache._hot_plan_reasoning_counter_seed = {0: (1.0, 2.0, 3.0, 4.0)}
    cache._hot_plan_counter_seed = {0: (4.0, 3.0, 2.0, 1.0)}
    cache.device = torch.device("cpu")
    cache.decayed_prefill_freq = None

    cache._apply_hot_plan_counter_seed()

    assert cache.decayed_reasoning_freq[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert cache.decayed_decode_freq[0].tolist() == [4.0, 3.0, 2.0, 1.0]


def _real_cache(monkeypatch):
    from freetoken.moe.host_banks import HostResidency

    OffloadMoeCache = _offload_cache_class_without_triton(monkeypatch)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=5, cache_size=7, device=torch.device("cpu"),
        prefill_overlap=False, decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    sources = {
        "gate_up": [torch.arange(5 * 4 * 3).view(5, 4, 3)],
        "down": [torch.arange(5 * 3 * 2).view(5, 3, 2)],
    }
    cache.set_bank_sources(
        sources, layer_residency=[HostResidency.DISK.value], hot_expert_ids={0: (1, 4)},
    )
    return cache, sum(b[0][0].numel() * b[0].element_size() for b in sources.values())


@pytest.mark.parametrize("histories", ["shared", "split", "split3"])
def test_configure_hot_adaptation_accepts_every_histories_value(monkeypatch, histories):
    """Regression: the real startup path (configure_hot_adaptation) rejected split3."""
    cache, row_bytes = _real_cache(monkeypatch)
    cache.configure_hot_adaptation(
        half_life_steps=2, interval_steps=1000,
        max_swap_bytes=row_bytes, expert_bytes=row_bytes,
        histories=histories, aim="phase",
        persisted_counter_seed={0: (1.0, 2.0, 3.0, 4.0, 5.0)},
        persisted_reasoning_counter_seed=(
            {0: (5.0, 4.0, 3.0, 2.0, 1.0)} if histories == "split3" else None
        ),
    )
    try:
        assert cache.hot_adapt_histories == histories
        assert (cache.decayed_prefill_freq is not None) == (histories != "shared")
        assert (cache.decayed_reasoning_freq is not None) == (histories == "split3")
        if histories == "split3":
            # Same storage as the stack the kernel indexes by phase.
            assert cache.decayed_reasoning_freq.data_ptr() == (
                cache.decayed_decode_freq.data_ptr()
                + cache.decayed_decode_freq.numel() * 4
            )
            assert cache.decayed_reasoning_freq[0].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]
            assert cache._hot_adapt_reasoning_snapshot_host.shape == (1, 5)
        assert cache.decayed_decode_freq[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    finally:
        cache.shutdown_hot_adaptation()


def test_configure_hot_adaptation_still_rejects_unknown_histories(monkeypatch):
    cache, row_bytes = _real_cache(monkeypatch)
    with pytest.raises(ValueError, match="split3"):
        cache.configure_hot_adaptation(
            half_life_steps=2, interval_steps=0,
            max_swap_bytes=row_bytes, expert_bytes=row_bytes, histories="split4",
        )


# --- production-visible logging ------------------------------------------------------


class _Log:
    def __init__(self):
        self.lines = []

    def info_rank0(self, message):
        self.lines.append(message)

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


def _plan_with_log(monkeypatch, histories, reasoning_phase, boundary="decode"):
    from freetoken.moe import hot_adapt

    monkeypatch.setattr(
        hot_adapt, "recompute_hot_partition", lambda *_a, **_k: {0: (0,)}
    )
    cache = _split3_cache(monkeypatch, histories=histories)
    log = _Log()
    monkeypatch.setitem(type(cache)._plan_hot_adaptation.__globals__, "logger", log)
    cache._hot_adapt_snapshot_host = torch.tensor([ANSWER[0]])
    cache._hot_adapt_prefill_snapshot_host = torch.tensor([PREFILL[0]])
    cache._hot_adapt_reasoning_snapshot_host = torch.tensor([REASONING[0]])
    cache._hot_adapt_tick_reasoning = reasoning_phase
    cache.hot_expert_capacity = {0: 1}
    cache.hot_adapt_aim = "phase"
    cache.hot_adapt_prefill_blend = 0.25
    cache.hot_adapt_expert_bytes = 1
    cache.num_experts = 4
    cache._hot_slot_owners = {0: [0]}
    cache.hot_adapt_max_swap_bytes = 1
    cache.hot_adapt_hot_budget_bytes = 1
    cache.hot_adapt_boundary_cap_frac = 1.0
    cache.hot_adapt_ticks_reasoning = 3
    cache.hot_adapt_ticks_answer = 4
    cache._plan_hot_adaptation(
        None, token=1, swap_budget_bytes=1, boundary=boundary, tick_count=1
    )
    return log.lines[-1]


def test_tick_line_carries_aim_and_per_history_ticks(monkeypatch):
    line = _plan_with_log(monkeypatch, "split3", True)
    assert "boundary=decode" in line
    assert "aim=reasoning" in line and "ticks_reasoning=3" in line
    assert "ticks_answer=4" in line and "reasoning_share=" in line
    assert "aim=answer" in _plan_with_log(monkeypatch, "split3", False)
    assert "aim=reasoning" in _plan_with_log(monkeypatch, "split3", True, "prefill")


def test_tick_line_for_split_has_no_split3_fields(monkeypatch):
    line = _plan_with_log(monkeypatch, "split", False)
    assert "aim=" not in line and "ticks_reasoning" not in line
    assert "decode_pair_rate=" in line


def test_decode_pair_rate_is_comparable_across_split_and_split3(monkeypatch):
    """Same traffic, hot row 0: split sees it as one decode history, split3 as two."""
    import re

    def decode_rate(line):
        return float(re.search(r"decode_pair_rate=([0-9.]+)%", line).group(1))

    from freetoken.moe import hot_adapt

    both = ANSWER[0]
    merged = tuple(a + r for a, r in zip(ANSWER[0], REASONING[0]))
    # split arm: one decode history holding answer + reasoning traffic.
    split_cache_line = None
    monkeypatch.setattr(
        hot_adapt, "recompute_hot_partition", lambda *_a, **_k: {0: (0,)}
    )
    cache = _split3_cache(monkeypatch, histories="split")
    log = _Log()
    monkeypatch.setitem(type(cache)._plan_hot_adaptation.__globals__, "logger", log)
    cache._hot_adapt_snapshot_host = torch.tensor([merged])
    cache._hot_adapt_prefill_snapshot_host = torch.tensor([PREFILL[0]])
    cache._hot_adapt_tick_reasoning = False
    cache.hot_expert_capacity = {0: 1}
    cache.hot_adapt_aim = "phase"
    cache.hot_adapt_prefill_blend = 0.25
    cache.hot_adapt_expert_bytes = 1
    cache.num_experts = 4
    cache._hot_slot_owners = {0: [0]}
    cache.hot_adapt_max_swap_bytes = 1
    cache.hot_adapt_hot_budget_bytes = 1
    cache.hot_adapt_boundary_cap_frac = 1.0
    cache._plan_hot_adaptation(
        None, token=1, swap_budget_bytes=1, boundary="decode", tick_count=1
    )
    split_cache_line = log.lines[-1]
    split3_line = _plan_with_log(monkeypatch, "split3", True)
    assert both  # row 0 holds 8 of 18 decode counts
    assert decode_rate(split_cache_line) == pytest.approx(decode_rate(split3_line))
    assert decode_rate(split3_line) == pytest.approx(100 * 8 / 18, abs=0.01)


def test_phase_flip_is_logged_and_rate_limited(monkeypatch):
    cache = _split3_cache(monkeypatch)
    log = _Log()
    monkeypatch.setitem(type(cache).set_decode_phase.__globals__, "logger", log)
    clock = iter([100.0, 100.2, 100.4, 102.0])
    monkeypatch.setattr(
        type(cache).set_decode_phase.__globals__["time"], "monotonic", lambda: next(clock)
    )
    for phase in (True, False, True, False):
        cache.set_decode_phase(phase)

    assert len(log.lines) == 2  # first flip, then the one after the 1 s window
    assert "-> reasoning" in log.lines[0]
    assert "-> answer" in log.lines[1] and "suppressed_since_last_log=2" in log.lines[1]
