from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import os

from freetoken.engine.cache_budget import (
    expert_bytes_per_slot,
    plan_cache_budget,
    required_bytes,
    resolve_moe_cache_auto,
)
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10 - 1  # one physical page is the dummy
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=910, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # The 10-page usable floor also charges one dummy: raw = (910 - 11*10) // 100 = 8.
    assert size == 8
    assert pages == 10
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10 - 1


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 400 B + 10 usable pages and one dummy = 510 B


def test_required_bytes_charges_one_dummy_beyond_reported_usable_pages():
    assert required_bytes(
        moe_cache_size=8,
        num_pages=10,
        per_expert_bytes=100,
        cache_per_page=10,
    ) == 910


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10 - 1


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once_and_marlin_cap():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1, quant_format="bf16",
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 39 usable pages + 1 dummy.
    assert size == 8 and pages == 39 and overlap is True


def test_resolve_auto_reserves_usable_tokens_beyond_the_dummy_page():
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=940,
        weights_bytes=0,
        memory_ratio=1.0,
        cache_per_page=10,
        fixed_cache_size=0,
        per_expert_bytes=100,
        num_experts=2,
        total_experts=50,
        prefill_overlap=False,
        kv_reserve_tokens=256,
        page_size=64,
        quant_format="bf16",
    )
    assert overlap is False
    assert pages * 64 >= 256
    assert size == 8 and pages == 13


def test_resident_mtp_head_is_charged_before_expert_cache_and_kv_planning():
    from freetoken.engine.engine import _startup_kv_budget, _state_dict_nbytes

    dense_bytes = 2_000
    mtp_state = {
        "experts.gate_up": torch.empty(300, dtype=torch.bfloat16),
        "experts.down": torch.empty(200, dtype=torch.bfloat16),
    }
    mtp_bytes = _state_dict_nbytes(mtp_state)
    assert mtp_bytes == 1_000

    baseline_free = 10_000
    post_weights_free = baseline_free - dense_bytes - mtp_bytes
    kv_budget = _startup_kv_budget(0.9, baseline_free, post_weights_free)
    assert kv_budget == int(0.9 * baseline_free) - dense_bytes - mtp_bytes

    # The same measured resident-weight delta feeds --moe-cache-auto before the KV
    # pool is allocated. Charging the MTP head removes exactly its bytes from the
    # joint expert-cache/KV budget.
    kwargs = dict(
        baseline_free=baseline_free,
        memory_ratio=0.9,
        cache_per_page=10,
        fixed_cache_size=0,
        per_expert_bytes=100,
        num_experts=2,
        total_experts=10,
        prefill_overlap=False,
        kv_reserve_tokens=0,
        page_size=1,
        quant_format="bf16",
    )
    _, pages_without_head, _ = resolve_moe_cache_auto(
        weights_bytes=dense_bytes, **kwargs
    )
    _, pages_with_head, _ = resolve_moe_cache_auto(
        weights_bytes=dense_bytes + mtp_bytes, **kwargs
    )
    assert pages_without_head * 10 - pages_with_head * 10 == mtp_bytes


def test_resolve_auto_marlin_caps_slots():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, quant_format="nvfp4_marlin",
    )
    assert size == 992


def test_qwen_flash_next_fp8_ladder_reference_geometry_is_consistent():
    from freetoken.attention import AttnType
    from freetoken.kernel.aot_models import expert_bank_row_bytes
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.scheduler.kv_ladder import KVLadderPolicy

    page_size = 64
    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=tuple(range(12)),
        num_kv_heads=2,
        head_dim=256,
        sliding_window=None,
        index_head_dim=128,
        num_index_layers=12,
        index_ratio=4,
        attn_type=AttnType.QSA,
    )
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_e4m3",
        page_size=page_size,
        max_running_req=1,
        speculative_mtp="off",
        tp_info=SimpleNamespace(size=1),
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: (spec,)),
    )
    kv_bytes_per_page, _, _, _ = QSAKVCache.kv_cost(config)
    expert_bytes = sum(expert_bank_row_bytes("nvfp4", 2560, 640).values())
    assert expert_bytes == 2_772_480
    assert kv_bytes_per_page == 13_056 * page_size == 835_584

    # Calibrate the otherwise unknown measured pool budget to the reported 3,753-slot
    # floor. This same budget reproduces the older 3,290-slot, 163,840-token reference.
    floor_tokens = 65_536
    floor_slots = 3_753
    budget = required_bytes(
        floor_slots,
        floor_tokens // page_size,
        expert_bytes,
        kv_bytes_per_page,
    )
    policy = KVLadderPolicy(
        step_tokens=32_768,
        max_context_tokens=100_352,
        page_size=page_size,
        pool_budget_bytes=budget,
        kv_bytes_per_page=kv_bytes_per_page,
        moe_bytes_per_slot=expert_bytes,
        min_moe_slots=1_024,
        prefill_overlap=True,
    )
    assert {
        tokens: policy.moe_slots_at_tokens(tokens, floor_slots)
        for tokens in (65_536, 98_304, 100_352, 163_840)
    } == {
        65_536: 3_753,
        98_304: 3_598,
        100_352: 3_589,
        163_840: 3_290,
    }


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        moe_disk_layers = None
        nvfp4_backend = "auto"
        moe_activation_dtype = "auto"
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_backend == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    # Keep this fixture pinned to fi. Never select auto or another backend to make
    # a machine without FlashInfer pass; dependency absence is an explicit skip.
    pytest.importorskip("flashinfer")
    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        nvfp4_backend = "auto"
        moe_activation_dtype = "auto"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16, quant_format="bf16",
    )
    assert (size, pages, overlap) == expected


def test_ladder_auto_plan_uses_two_steps_as_floor_and_explicit_pages_as_cap(
    monkeypatch,
):
    from freetoken.engine import cache_budget, engine as engine_module
    from freetoken.engine.engine import Engine

    class Pool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 64, 0

    class ModelConfig:
        dsv4_args = None
        is_moe = True
        num_experts = 128
        num_moe_layers = 64
        slot_states = ()

        @staticmethod
        def linear_attention_group():
            return None

    config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_backend="offload",
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        num_page_override=1568,
        max_seq_len=100_352,
        kv_reserve_tokens=8192,
        memory_ratio=0.9,
        moe_prefill_overlap=True,
        model_config=ModelConfig(),
    )
    banks = SimpleNamespace(
        quant_format="bf16",
        sources={"gate_up": [torch.zeros(1, 25, dtype=torch.float32)]},
    )
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return 3907, 2000, True

    logs = []
    monkeypatch.setattr(cache_budget, "resolve_moe_cache_auto", fake_resolve)
    monkeypatch.setattr(
        engine_module.logger,
        "info_rank0",
        lambda message, *args: logs.append(message % args),
    )
    engine = Engine.__new__(Engine)
    engine._pool_cls = Pool
    engine._baseline_free = 10_000
    engine._weights_bytes = 100

    size, pages, overlap = engine._resolve_auto_moe_cache_size(config, banks)

    assert (size, pages, overlap) == (3907, 1568, True)
    assert captured["kv_reserve_tokens"] == 65_536
    assert config.kv_ladder_floor_tokens == 65_536
    assert config.kv_ladder_cap_tokens == 100_352
    assert config.kv_ladder_explicit_cap is True
    assert any("raised floor from 8192 to 65536 tokens" in line for line in logs)


def test_ladder_warns_and_preserves_pool_minimum_above_cap(monkeypatch):
    from freetoken.engine import cache_budget, engine as engine_module
    from freetoken.engine.engine import Engine

    pool_minimum = 131_072

    class Pool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 64, pool_minimum

    config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_backend="offload",
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        num_page_override=1568,
        max_seq_len=100_352,
        kv_reserve_tokens=8192,
        memory_ratio=0.9,
        moe_prefill_overlap=True,
        model_config=SimpleNamespace(
            dsv4_args=None,
            is_moe=True,
            num_experts=128,
            num_moe_layers=64,
            slot_states=(),
            linear_attention_group=lambda: None,
        ),
    )
    banks = SimpleNamespace(
        quant_format="bf16",
        sources={"gate_up": [torch.zeros(1, 25, dtype=torch.float32)]},
    )
    captured = {}
    warnings = []

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return 3907, 3000, True

    monkeypatch.setattr(cache_budget, "resolve_moe_cache_auto", fake_resolve)
    monkeypatch.setattr(
        engine_module.logger,
        "warning_rank0",
        lambda message, *args: warnings.append(message % args),
    )
    engine = Engine.__new__(Engine)
    engine._pool_cls = Pool
    engine._baseline_free = 10_000
    engine._weights_bytes = 100

    size, pages, overlap = engine._resolve_auto_moe_cache_size(config, banks)

    assert (size, pages, overlap) == (3907, pool_minimum // 64, True)
    assert captured["kv_reserve_tokens"] == pool_minimum
    assert config.kv_ladder_floor_tokens == pool_minimum
    assert any(
        "pool minimum 131072 tokens is above KV ladder cap 100352" in line
        for line in warnings
    )


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def test_engine_config_declares_derived_ladder_geometry_fields():
    from dataclasses import fields

    from freetoken.engine.config import EngineConfig

    ladder_fields = {
        item.name: item
        for item in fields(EngineConfig)
        if item.name.startswith("kv_ladder_")
    }
    assert ladder_fields["kv_ladder_floor_tokens"].init is False
    assert ladder_fields["kv_ladder_cap_tokens"].init is False
    assert ladder_fields["kv_ladder_explicit_cap"].init is False


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_backend="auto") unless overridden, shared by the _adjust_config tests."""
    # Keep this fixture pinned to fi. Never select auto or another backend to make
    # a machine without FlashInfer pass; dependency absence is an explicit skip.
    pytest.importorskip("flashinfer")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",  # quantized experts -> must resolve to an offload backend
            moe_backend="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend():
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_backend="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_backend

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
    assert is_offload_moe_backend(config.moe_backend)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


def test_adjust_config_rejects_explicit_ladder_after_auto_backend_resolution():
    from freetoken.engine.engine import _adjust_config

    config = _offload_engine_config(
        kv_ladder="on",
        kv_ladder_explicit=True,
        max_running_req=4,
    )
    assert config.moe_backend == "auto"
    assert config.moe_cache_auto is False

    with pytest.raises(ValueError, match="requires --max-running-requests 1"):
        _adjust_config(config)


def test_auto_pages_replace_explicit_override_only_for_active_ladder():
    from freetoken.engine.engine import _install_auto_moe_cache_pages

    model_config = SimpleNamespace(dsv4_args=None, is_moe=True)
    active = SimpleNamespace(
        num_page_override=1568,
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_backend="offload",
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        model_config=model_config,
    )
    fixed = SimpleNamespace(
        num_page_override=1568,
        kv_ladder="off",
        moe_cache_auto=True,
    )

    _install_auto_moe_cache_pages(active, 1025)
    _install_auto_moe_cache_pages(fixed, 1025)

    assert active.num_page_override == 1025
    assert fixed.num_page_override == 1568


def test_ineligible_tp_ladder_keeps_explicit_pages_and_fixed_reserve(monkeypatch):
    from freetoken.engine import cache_budget
    from freetoken.engine.engine import Engine, _install_auto_moe_cache_pages

    class Pool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 64, 0

    config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_backend="offload",
        max_running_req=1,
        tp_info=SimpleNamespace(size=2),
        num_page_override=1568,
        max_seq_len=100_352,
        kv_reserve_tokens=8192,
        memory_ratio=0.9,
        moe_prefill_overlap=True,
        model_config=SimpleNamespace(
            dsv4_args=None,
            is_moe=True,
            num_experts=128,
            num_moe_layers=64,
            slot_states=(),
            linear_attention_group=lambda: None,
        ),
    )
    banks = SimpleNamespace(
        quant_format="bf16",
        sources={"gate_up": [torch.zeros(1, 25, dtype=torch.float32)]},
    )
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return 3907, 1025, True

    monkeypatch.setattr(cache_budget, "resolve_moe_cache_auto", fake_resolve)
    engine = Engine.__new__(Engine)
    engine._pool_cls = Pool
    engine._baseline_free = 10_000
    engine._weights_bytes = 100

    assert engine._resolve_auto_moe_cache_size(config, banks) == (3907, 1025, True)
    assert captured["kv_reserve_tokens"] == 8192
    assert not hasattr(config, "kv_ladder_cap_tokens")

    _install_auto_moe_cache_pages(config, 1025)
    assert config.num_page_override == 1568


def test_ineligible_dsv4_ladder_preserves_pool_minimum_reserve(monkeypatch):
    from freetoken.engine import cache_budget
    from freetoken.engine.engine import Engine

    class Pool:
        @staticmethod
        def kv_cost(config):
            return 10, 0, 128, 131_072

    config = SimpleNamespace(
        kv_ladder="on",
        moe_cache_auto=True,
        moe_cache_size=0,
        moe_cache_rate=None,
        moe_backend="offload",
        max_running_req=1,
        tp_info=SimpleNamespace(size=1),
        num_page_override=8,
        max_seq_len=100_352,
        kv_reserve_tokens=8192,
        memory_ratio=0.9,
        moe_prefill_overlap=True,
        model_config=SimpleNamespace(
            dsv4_args=object(),
            is_moe=True,
            num_experts=128,
            num_moe_layers=64,
            slot_states=(),
            linear_attention_group=lambda: None,
        ),
    )
    banks = SimpleNamespace(
        quant_format="bf16",
        sources={"gate_up": [torch.zeros(1, 25, dtype=torch.float32)]},
    )
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return 3907, 1025, True

    monkeypatch.setattr(cache_budget, "resolve_moe_cache_auto", fake_resolve)
    engine = Engine.__new__(Engine)
    engine._pool_cls = Pool
    engine._baseline_free = 10_000
    engine._weights_bytes = 100

    engine._resolve_auto_moe_cache_size(config, banks)

    assert captured["kv_reserve_tokens"] == 131_072
    assert not hasattr(config, "kv_ladder_cap_tokens")


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override):
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=SimpleNamespace(max_position=max_position),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        nvfp4_backend = "auto"
        moe_activation_dtype = "auto"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None
