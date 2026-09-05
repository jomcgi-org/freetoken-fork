from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import UserMsg
from freetoken.scheduler.kv_ladder import (
    KVLadderCapacityError,
    KVLadderPolicy,
    KVLadderProtectedCapacityError,
    kv_ladder_requested,
)


def _policy(*, kv_bytes_per_page: int, budget: int, protected=(), overlap=False):
    return KVLadderPolicy(
        step_tokens=32,
        max_context_tokens=256,
        page_size=1,
        pool_budget_bytes=budget,
        kv_bytes_per_page=kv_bytes_per_page,
        moe_bytes_per_slot=100,
        min_moe_slots=4,
        prefill_overlap=overlap,
        protected_rows_by_layer=protected,
    )


@pytest.mark.parametrize(
    ("kv_cache_dtype", "current_slots", "target_slots"),
    [("bf16", 93, 87), ("fp8_e4m3", 96, 93)],
    ids=["bf16-pool", "fp8-pool"],
)
def test_ladder_prices_the_pool_storage_dtype(
    kv_cache_dtype, current_slots, target_slots
):
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.kvcache.mha_pool import MHAKVCache

    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=(0,),
        num_kv_heads=1,
        head_dim=5,
        sliding_window=None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: (spec,)),
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        page_size=1,
        tp_info=SimpleNamespace(size=1),
    )
    kv_bytes_per_page, _, _, _ = MHAKVCache.kv_cost(config)
    plan = _policy(kv_bytes_per_page=kv_bytes_per_page, budget=10_000).plan(
        current_pages=32,
        current_moe_slots=current_slots,
        input_tokens=33,
        max_output_tokens=1,
    )
    assert plan is not None
    assert plan.target_pages == 64
    assert plan.target_moe_slots == target_slots


def test_growth_triggers_only_when_request_exceeds_current_pool():
    policy = _policy(kv_bytes_per_page=10, budget=10_000)
    assert policy.plan(
        current_pages=32,
        current_moe_slots=96,
        input_tokens=16,
        max_output_tokens=16,
    ) is None
    plan = policy.plan(
        current_pages=32,
        current_moe_slots=96,
        input_tokens=16,
        max_output_tokens=17,
    )
    assert plan is not None
    assert plan.target_tokens == 64


def test_next_growth_stays_aligned_when_startup_charged_a_dummy_page():
    policy = _policy(kv_bytes_per_page=10, budget=10_000)
    plan = policy.plan(
        current_pages=33,
        current_moe_slots=96,
        input_tokens=34,
        max_output_tokens=1,
    )
    assert plan is not None
    assert plan.current_tokens == 33
    assert plan.target_tokens == 64


def test_ladder_budget_includes_the_dummy_page():
    policy = _policy(kv_bytes_per_page=10, budget=1_049)
    with pytest.raises(KVLadderCapacityError, match="minimum 4 MoE slots"):
        policy.plan(
            current_pages=32,
            current_moe_slots=4,
            input_tokens=33,
            max_output_tokens=1,
        )


def test_protected_rows_are_preserved_when_the_target_has_room():
    plan = _policy(
        kv_bytes_per_page=10,
        budget=1_950,
        protected=((3, 4), (7, 4)),
    ).plan(
        current_pages=32,
        current_moe_slots=20,
        input_tokens=33,
        max_output_tokens=1,
    )
    assert plan is not None
    assert plan.target_moe_slots == 13
    assert plan.target_moe_slots >= 8 + 4


def test_protected_rows_refuse_growth_when_fetch_reserve_would_be_consumed():
    with pytest.raises(KVLadderProtectedCapacityError) as raised:
        _policy(
            kv_bytes_per_page=10,
            budget=1_650,
            protected=((3, 4), (7, 4)),
        ).plan(
            current_pages=32,
            current_moe_slots=20,
            input_tokens=33,
            max_output_tokens=1,
        )

    assert raised.value.protected == 8
    assert raised.value.available_non_protected == 8
    assert raised.value.required == 10


def test_protected_capacity_drain_is_terminal_and_does_not_block_following_request(
    monkeypatch,
):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=1_650)
    blocked = UserMsg(
        uid=17,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
        arrival_time=0.0,
    )
    following = UserMsg(
        uid=18,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
        arrival_time=31.0,
    )
    scheduler._kv_ladder_waiting = [blocked]
    scheduler._kv_ladder_starvation_uid = None
    scheduler._abort_tombstones = {}
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=SimpleNamespace(
            cache_size=20,
            hot_expert_capacity={3: 4, 7: 4},
            prefill_overlap=False,
        )
    )
    scheduler.prefill_manager = SimpleNamespace(
        runnable=False,
        pending_list=[],
        add_one_req=lambda request: admitted.append(request.uid),
        clock=lambda: 31.0,
        priority_aging_seconds=30.0,
    )
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)
    admitted = []
    logs = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning_rank0",
        lambda message, *args: logs.append(message % args),
    )

    Scheduler._process_one_msg(scheduler, following)
    assert admitted == [18]
    assert scheduler._kv_ladder_waiting == [blocked]

    Scheduler._drain_kv_ladder_waiting(scheduler)

    assert admitted == [18, 17]
    assert blocked.sampling_params.max_tokens == 16
    assert scheduler._kv_ladder_waiting == []
    assert logs[-2] == (
        "KV ladder cannot grow for request 17: "
        "KV ladder rung 64 tokens would evict protected HOT rows: protected=8, "
        "available_non_protected=8, required=10"
    )


def test_flag_and_auto_sizing_gate_the_policy():
    def config(**overrides):
        values = dict(
            kv_ladder="on",
            moe_cache_auto=True,
            moe_cache_size=0,
            moe_cache_rate=None,
            max_running_req=1,
            tp_info=SimpleNamespace(size=1),
            moe_backend="offload",
            model_config=SimpleNamespace(dsv4_args=None, is_moe=True),
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    assert kv_ladder_requested(config())
    assert not kv_ladder_requested(
        config(max_running_req=4)
    )
    assert not kv_ladder_requested(config(kv_ladder="off"))
    assert not kv_ladder_requested(config(moe_cache_auto=False))
    assert not kv_ladder_requested(config(tp_info=SimpleNamespace(size=2)))
    assert not kv_ladder_requested(
        config(model_config=SimpleNamespace(dsv4_args=object(), is_moe=True))
    )
    assert not kv_ladder_requested(config(moe_backend="fused"))

    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(kv_ladder="off", moe_cache_auto=True)
    assert Scheduler._make_kv_ladder_policy(scheduler) is None


def test_hot_adaptation_is_drained_before_protected_geometry_changes():
    from freetoken.engine.engine import Engine

    events = []

    class Future:
        def cancel(self):
            events.append("cancel")
            return False

        def result(self):
            events.append("drain")

    cache = SimpleNamespace(
        _hot_adapt_future=Future(),
        _hot_adapt_phase="copy",
        _hot_adapt_swaps_pending=(object(),),
        _hot_adapt_worker_installs=True,
    )

    Engine._drain_hot_adaptation_for_rebuild(cache)

    assert events == ["cancel", "drain"]
    assert cache._hot_adapt_future is None
    assert cache._hot_adapt_phase is None
    assert cache._hot_adapt_swaps_pending == ()
    assert cache._hot_adapt_worker_installs is False


def test_scheduler_holds_growth_request_before_admission():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    scheduler._kv_ladder_waiting = []
    scheduler._abort_tombstones = {}
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=SimpleNamespace(
            cache_size=96,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    admitted = []
    scheduler.prefill_manager = SimpleNamespace(add_one_req=admitted.append)
    msg = UserMsg(
        uid=5,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
    )

    Scheduler._process_one_msg(scheduler, msg)

    assert admitted == []
    assert scheduler._kv_ladder_waiting == [msg]


def test_non_growing_request_bypasses_parked_ladder_request():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    parked = UserMsg(
        uid=1,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
    )
    scheduler._kv_ladder_waiting = [parked]
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        moe_offload_cache=SimpleNamespace(
            cache_size=96,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    scheduler.prefill_manager = SimpleNamespace()
    fitting = UserMsg(
        uid=2,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
        priority=5,
    )

    assert not Scheduler._queue_for_kv_ladder(scheduler, fitting)
    assert scheduler._kv_ladder_waiting == [parked]


def test_higher_priority_growth_request_overtakes_parked_request():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    scheduler._kv_ladder_waiting = []
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        moe_offload_cache=SimpleNamespace(
            cache_size=96,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    scheduler.prefill_manager = SimpleNamespace(
        clock=lambda: 20.0,
        priority_aging_seconds=30.0,
    )
    low = UserMsg(
        uid=1,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
        priority=0,
        arrival_time=10.0,
    )
    high = UserMsg(
        uid=2,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=49),
        priority=5,
        arrival_time=20.0,
    )

    assert Scheduler._queue_for_kv_ladder(scheduler, low)
    assert Scheduler._queue_for_kv_ladder(scheduler, high)
    assert scheduler._kv_ladder_waiting == [high, low]


def test_starvation_bound_holds_new_admissions_and_drains_aged_waiter_first(
    monkeypatch,
):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    scheduler._kv_ladder_starvation_uid = None
    scheduler._abort_tombstones = {}
    aged = UserMsg(
        uid=1,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
        priority=-100,
        arrival_time=0.0,
    )
    newcomer = UserMsg(
        uid=2,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=16),
        priority=100,
        arrival_time=31.0,
    )
    scheduler._kv_ladder_waiting = [aged]
    moe = SimpleNamespace(
        cache_size=96,
        hot_expert_capacity={},
        prefill_overlap=False,
        _hot_slot_owners={},
    )
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=moe,
    )
    events = []
    scheduler.prefill_manager = SimpleNamespace(
        runnable=False,
        pending_list=[],
        priority_aging_seconds=30.0,
        clock=lambda: 31.0,
        add_one_req=lambda request: events.append(("admit", request.uid)),
    )
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)

    def execute(**kwargs):
        events.append(("rebuild", scheduler._pending_rebuild.num_pages))
        scheduler.engine.num_pages = scheduler._pending_rebuild.num_pages
        scheduler.engine.max_seq_len = scheduler._pending_rebuild.num_pages
        moe.cache_size = scheduler._pending_rebuild.moe_cache_size
        scheduler._pending_rebuild = None
        return "ok"

    scheduler._execute_pending_rebuild = execute
    logs = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning_rank0",
        lambda message, *args: logs.append(message % args),
    )

    Scheduler._process_one_msg(scheduler, newcomer)

    assert events == []
    assert scheduler._kv_ladder_waiting == [newcomer, aged]
    assert logs == [
        "KV ladder starvation bound reached: request 1 waited 31.0s "
        "(limit 30.0s); pausing new admissions until it is grown and admitted"
    ]

    Scheduler._drain_kv_ladder_waiting(scheduler)

    assert events == [("rebuild", 64), ("admit", 1)]
    assert scheduler._kv_ladder_waiting == [newcomer]
    assert scheduler._kv_ladder_starvation_uid is None


def test_queue_stats_include_ladder_waiters():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_manager = SimpleNamespace(
        pending_list=[SimpleNamespace(priority=0, arrival_time=90.0)],
        clock=lambda: 100.0,
    )
    scheduler._kv_ladder_waiting = [
        SimpleNamespace(priority=4, arrival_time=80.0),
    ]

    depth, bands, max_wait = Scheduler._queue_stats(scheduler)

    assert depth == 2
    assert bands == {"negative": 0, "zero": 1, "positive": 1}
    assert max_wait == 20.0


def test_scheduler_clamps_default_output_budget_to_the_ladder_cap():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    scheduler.engine = SimpleNamespace(
        num_pages=224,
        moe_offload_cache=SimpleNamespace(
            cache_size=70,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    msg = UserMsg(
        uid=7,
        input_ids=torch.arange(240, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=32_768),
    )

    plan = Scheduler._kv_ladder_plan(scheduler, msg)

    assert plan is not None
    assert plan.required_tokens == 256
    assert plan.target_tokens == 256


def test_default_output_budget_does_not_grow_the_64k_floor_for_a_short_prompt():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = KVLadderPolicy(
        step_tokens=32_768,
        max_context_tokens=100_352,
        page_size=64,
        pool_budget_bytes=1_000_000,
        kv_bytes_per_page=10,
        moe_bytes_per_slot=100,
        min_moe_slots=4,
        prefill_overlap=False,
    )
    scheduler.engine = SimpleNamespace(
        num_pages=1025,
        moe_offload_cache=SimpleNamespace(
            cache_size=100,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    msg = UserMsg(
        uid=8,
        input_ids=torch.arange(64, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=32_768),
    )

    assert Scheduler._kv_ladder_plan(scheduler, msg) is None


def _scheduler_for_ladder_gate(parsed):
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        kv_ladder=parsed.kv_ladder,
        kv_ladder_explicit=parsed.kv_ladder_explicit,
        moe_cache_auto=parsed.moe_cache_auto,
        moe_cache_size=parsed.moe_cache_size,
        moe_cache_rate=parsed.moe_cache_rate,
        max_running_req=parsed.max_running_req,
        tp_info=SimpleNamespace(size=1),
        moe_backend=parsed.moe_backend,
        model_config=SimpleNamespace(dsv4_args=None, is_moe=True),
    )
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)
    scheduler.engine = SimpleNamespace(moe_offload_cache=SimpleNamespace())
    return scheduler


def test_default_ladder_with_default_concurrency_is_inert_not_fatal(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.server.args import parse_args

    parsed, _ = parse_args(
        ["--model", "/models/anon", "--dtype", "bfloat16", "--moe-backend", "offload"]
    )
    assert parsed.kv_ladder == "on"
    assert parsed.kv_ladder_explicit is False
    assert parsed.moe_cache_auto is True
    assert parsed.max_running_req == 4
    logs = []
    monkeypatch.setattr(scheduler_module.logger, "warning_rank0", logs.append)

    assert Scheduler._make_kv_ladder_policy(_scheduler_for_ladder_gate(parsed)) is None
    assert logs == ["KV ladder inactive: --max-running-requests must be 1"]


def test_ladder_policy_consults_cache_manager_rebuild_gate(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        moe_backend="offload",
        model_config=SimpleNamespace(dsv4_args=None, is_moe=True),
    )
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=False)
    logs = []
    monkeypatch.setattr(scheduler_module.logger, "warning_rank0", logs.append)

    assert Scheduler._make_kv_ladder_policy(scheduler) is None
    assert logs == [
        "KV ladder inactive: this model's cache does not support runtime rebuild"
    ]


def test_explicit_ladder_rejects_default_concurrency_during_argument_parsing():
    from freetoken.server.args import parse_args

    with pytest.raises(ValueError, match="requires --max-running-requests 1"):
        parse_args(
            [
                "--model",
                "/models/anon",
                "--dtype",
                "bfloat16",
                "--moe-backend",
                "offload",
                "--kv-ladder",
                "on",
            ]
        )


def test_explicit_ladder_with_fixed_cache_is_inert_not_fatal(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.server.args import parse_args

    parsed, _ = parse_args(
        [
            "--model",
            "/models/anon",
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "offload",
            "--moe-cache-size",
            "512",
            "--kv-ladder",
            "on",
        ]
    )
    logs = []
    monkeypatch.setattr(scheduler_module.logger, "warning_rank0", logs.append)

    assert Scheduler._make_kv_ladder_policy(_scheduler_for_ladder_gate(parsed)) is None
    assert "--moe-cache-auto is required" in logs[0]


@pytest.mark.parametrize(
    ("cache_flag", "sizing_flag"),
    [
        (("--moe-cache-size", "512"), "--moe-cache-size"),
        (("--moe-cache-rate", "0.5"), "--moe-cache-rate"),
    ],
)
def test_default_ladder_logs_fixed_cache_sizing_as_inert(
    monkeypatch, cache_flag, sizing_flag
):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.server.args import parse_args

    parsed, _ = parse_args(
        [
            "--model",
            "/models/anon",
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "offload",
            "--max-running-requests",
            "1",
            *cache_flag,
        ]
    )
    logs = []
    monkeypatch.setattr(scheduler_module.logger, "warning_rank0", logs.append)

    assert Scheduler._make_kv_ladder_policy(_scheduler_for_ladder_gate(parsed)) is None
    assert logs == [
        f"KV ladder inactive: MoE cache sizing uses {sizing_flag}; "
        "--moe-cache-auto is required"
    ]


def test_ladder_logs_inert_pool_and_dummy_page(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    class KVPool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 1, 0

    logs = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info_rank0",
        lambda message, *args: logs.append(message % args),
    )
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning_rank0",
        lambda message, *args: logs.append(message % args),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        moe_backend="offload",
        model_config=SimpleNamespace(
            dsv4_args=None,
            is_moe=True,
            num_experts=4,
            linear_attention_group=lambda: None,
            slot_states=(),
        ),
        max_seq_len=256,
        kv_ladder_cap_tokens=32,
        kv_ladder_floor_tokens=32,
        kv_ladder_explicit_cap=True,
        kv_reserve_tokens=8,
        memory_ratio=1.0,
    )
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)
    scheduler.engine = SimpleNamespace(
        kv_cache=KVPool(),
        linear_state_pool=None,
        num_pages=32,
        _baseline_free=10_000,
        _weights_bytes=0,
        moe_offload_cache=SimpleNamespace(
            cache_size=90,
            hot_expert_capacity={},
            prefill_overlap=False,
            bank_sources={
                "gate_up": [torch.zeros(1, 25, dtype=torch.float32)],
            },
        ),
    )

    Scheduler._make_kv_ladder_policy(scheduler)

    assert any("+1 dummy page" in line for line in logs)
    assert any("expert_slots_at_startup=90" in line for line in logs)
    assert not any("expert_slots_at_floor" in line for line in logs)
    assert any(
        "ladder inert: pool already at cap 32" in line
        and "configured --num-pages/--num-tokens cap" in line
        for line in logs
    )


def test_ladder_startup_warns_when_hot_budget_blocks_the_cap(monkeypatch):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    class KVPool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 1, 0

    logs = []
    monkeypatch.setattr(scheduler_module.logger, "info_rank0", lambda *_args: None)
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning_rank0",
        lambda message, *args: logs.append(message % args),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        moe_backend="offload",
        model_config=SimpleNamespace(
            dsv4_args=None,
            is_moe=True,
            num_experts=4,
            linear_attention_group=lambda: None,
            slot_states=(),
        ),
        max_seq_len=256,
        kv_ladder_cap_tokens=256,
        kv_ladder_floor_tokens=32,
        kv_ladder_explicit_cap=False,
        kv_reserve_tokens=8,
        memory_ratio=1.0,
    )
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)
    scheduler.engine = SimpleNamespace(
        kv_cache=KVPool(),
        linear_state_pool=None,
        num_pages=32,
        _baseline_free=10_000,
        _weights_bytes=0,
        moe_offload_cache=SimpleNamespace(
            cache_size=90,
            num_experts=4,
            hot_expert_capacity={layer: 4 for layer in range(18)},
            prefill_overlap=False,
            bank_sources={
                "gate_up": [torch.zeros(1, 25, dtype=torch.float32)],
            },
        ),
    )

    Scheduler._make_kv_ladder_policy(scheduler)

    assert logs == [
        "KV ladder cap cannot be reached without protected HOT eviction: "
        "slots=90, protected=72, fetch_reserve=4, needed=16, reclaimable=14; "
        "the first request that needs this capacity will use the current KV pool "
        "and be clamped, or receive context_length_exceeded when its prompt alone "
        "does not fit"
    ]


def test_idle_drain_rebuilds_before_admitting_the_request():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    msg = UserMsg(
        uid=6,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
    )
    scheduler._kv_ladder_waiting = [msg]
    moe = SimpleNamespace(
        cache_size=96,
        hot_expert_capacity={},
        prefill_overlap=False,
        _hot_slot_owners={},
    )
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=moe,
    )
    events = []
    scheduler.prefill_manager = SimpleNamespace(
        runnable=False,
        add_one_req=lambda request: events.append(("admit", request.uid)),
    )
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)

    def execute(**kwargs):
        events.append(("rebuild", scheduler._pending_rebuild.num_pages))
        scheduler.engine.num_pages = scheduler._pending_rebuild.num_pages
        scheduler.engine.max_seq_len = scheduler._pending_rebuild.num_pages
        moe.cache_size = scheduler._pending_rebuild.moe_cache_size
        scheduler._pending_rebuild = None
        return "ok"

    scheduler._execute_pending_rebuild = execute

    Scheduler._drain_kv_ladder_waiting(scheduler)

    assert events == [("rebuild", 64), ("admit", 6)]
    assert scheduler._kv_ladder_waiting == []


@pytest.mark.parametrize(
    ("cpu_executor", "collect_stats", "expected_hot_rate"),
    [(object(), True, "94.00%"), (None, True, "n/a"), (object(), False, "n/a")],
)
def test_growth_status_line_exposes_preserved_hot_rate_and_ple_faults(
    monkeypatch, cpu_executor, collect_stats, expected_hot_rate
):
    from freetoken.scheduler import scheduler as scheduler_module
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=1_950)
    msg = UserMsg(
        uid=18,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
    )
    scheduler._kv_ladder_waiting = [msg]
    scheduler._kv_ladder_starvation_uid = None
    def diagnostic(value):
        assert collect_stats, "disabled diagnostics must not read GPU/PLE statistics"
        return value

    moe = SimpleNamespace(
        cache_size=20,
        hot_expert_capacity={3: 4, 7: 4},
        prefill_overlap=False,
        cpu_executor=cpu_executor,
        collect_stats=collect_stats,
        disk_prefetch_stats=lambda **_kwargs: diagnostic({"hot_pair_rate": 0.94}),
    )
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=moe,
        model=SimpleNamespace(
            ple_disk_stats=lambda **_kwargs: diagnostic({"ple_major_faults": 400_000})
        ),
    )
    events = []
    scheduler.prefill_manager = SimpleNamespace(
        runnable=False,
        add_one_req=lambda request: events.append(("admit", request.uid)),
    )
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=True)

    def execute(**kwargs):
        events.append(("rebuild", kwargs))
        scheduler.engine.num_pages = scheduler._pending_rebuild.num_pages
        scheduler.engine.max_seq_len = scheduler._pending_rebuild.num_pages
        moe.cache_size = scheduler._pending_rebuild.moe_cache_size
        scheduler._pending_rebuild = None
        return "ok"

    scheduler._execute_pending_rebuild = execute
    logs = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info_rank0",
        lambda message, *args: logs.append(message % args),
    )

    Scheduler._drain_kv_ladder_waiting(scheduler)

    assert events == [
        ("rebuild", {"preserve_hot_state": True, "send_reply": False}),
        ("admit", 18),
    ]
    assert "protected_rows=8" in logs[-1]
    assert f"realized_hot_rate={expected_hot_rate}" in logs[-1]
    assert f"ple_major_faults={'400000' if collect_stats else 'n/a'}" in logs[-1]


def test_idle_drain_rechecks_cache_manager_rebuild_gate():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kv_ladder = _policy(kv_bytes_per_page=10, budget=10_000)
    msg = UserMsg(
        uid=6,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=17),
    )
    scheduler._kv_ladder_waiting = [msg]
    scheduler.engine = SimpleNamespace(
        num_pages=32,
        max_seq_len=32,
        moe_offload_cache=SimpleNamespace(
            cache_size=96,
            hot_expert_capacity={},
            prefill_overlap=False,
        ),
    )
    events = []
    scheduler.prefill_manager = SimpleNamespace(
        runnable=False,
        add_one_req=lambda request: events.append(("admit", request.uid)),
    )
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler.cache_manager = SimpleNamespace(supports_runtime_rebuild=False)
    scheduler._execute_pending_rebuild = lambda **kwargs: events.append(("rebuild", None))

    Scheduler._drain_kv_ladder_waiting(scheduler)

    assert events == [("admit", 6)]
    assert scheduler._kv_ladder_waiting == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_ladder_rebuild_preserves_protected_hot_rows_and_realized_hits():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=8,
        device=torch.device("cuda"),
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    sources = {
        "gate_up": [torch.arange(4 * 8 * 4).view(4, 8, 4)],
        "down": [torch.arange(4 * 4 * 2).view(4, 4, 2)],
    }
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 3)},
    )
    row_bytes = sum(
        bank[0][0].numel() * bank[0].element_size()
        for bank in sources.values()
    )
    cache.configure_hot_adaptation(
        half_life_steps=2,
        interval_steps=0,
        max_swap_bytes=row_bytes,
        expert_bytes=row_bytes,
    )
    try:
        cache.stat_hot_pairs.fill_(94)
        cache.stat_hot_total_pairs.fill_(100)
        cache.decayed_decode_freq.copy_(
            torch.tensor([[1.0, 7.0, 2.0, 9.0]], device="cuda")
        )
        cache.decode_freq.copy_(
            torch.tensor([[1, 7, 2, 9]], dtype=torch.int64, device="cuda")
        )
        cache._protected_route_baseline = [[0, 3, 0, 4]]
        owners_before = {
            layer_id: tuple(owners)
            for layer_id, owners in cache._hot_slot_owners.items()
        }
        plan_before = dict(cache._hot_plan_last_published_owners)
        decayed_before = cache.decayed_decode_freq.clone()
        decode_before = cache.decode_freq.clone()

        cache.rebuild(6, preserve_hot_state=True)

        assert {
            layer_id: tuple(owners)
            for layer_id, owners in cache._hot_slot_owners.items()
        } == owners_before
        for expert in (1, 3):
            slot = int(cache.slot_for_id[0, expert].item())
            assert slot >= 0
            for name in cache.bank_schema:
                assert torch.equal(
                    cache.bank_caches[name][slot].cpu(),
                    sources[name][0][expert],
                )
        assert int(cache.stat_hot_pairs.item()) == 94
        assert int(cache.stat_hot_total_pairs.item()) == 100
        assert (
            int(cache.stat_hot_pairs.item())
            / int(cache.stat_hot_total_pairs.item())
            == 0.94
        )
        assert torch.equal(cache.decayed_decode_freq, decayed_before)
        assert torch.equal(cache.decode_freq, decode_before)
        assert cache._protected_route_baseline == [[0, 3, 0, 4]]
        assert cache._hot_plan_last_published_owners == plan_before
    finally:
        cache.shutdown_hot_adaptation()
