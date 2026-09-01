"""GPU-free GLM coverage for the shared routed-expert disk tier seams."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _config(backend: str = "offload") -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=5,
        first_k_dense_replace=2,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=16,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        dense_quant="none",
        moe_backend=backend,
    )


@pytest.mark.parametrize(
    "block_cls",
    [
        pytest.param(
            "freetoken.models.glm4_moe.moe.Glm4MoeSparseBlock", id="glm4_moe"
        ),
        pytest.param(
            "freetoken.models.glm_moe_dsa.moe.GlmMoeDsaSparseBlock",
            id="glm_moe_dsa",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["offload", "fused"])
def test_glm_sparse_block_selects_configured_expert_path(monkeypatch, block_cls, backend):
    from freetoken.layers import MoELayer, OffloadMoELayer

    module_name, class_name = block_cls.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    _init_tp()
    block = cls(_config(backend), layer_id=2)

    if backend == "offload":
        assert isinstance(block.experts, OffloadMoELayer)
    else:
        assert type(block.experts) is MoELayer

    calls = []

    def routed(hidden, weights, ids):
        calls.append((weights.clone(), ids.clone()))
        return torch.full_like(hidden, 2.0)

    monkeypatch.setattr(block.experts, "routed_forward", routed)
    monkeypatch.setattr(
        block.shared_experts, "forward", lambda hidden: torch.ones_like(hidden)
    )
    with torch.no_grad():
        block.gate.weight.zero_()
        block.e_score_correction_bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    out = block.forward(torch.zeros(2, 16))

    assert len(calls) == 1
    assert calls[0][1].tolist() == [[0, 1], [0, 1]]
    assert torch.equal(out, torch.full((2, 16), 3.0))


def test_glm_bank_geometry_comes_from_synthetic_config():
    from freetoken.models.glm_moe import (
        GlmMoeGeometry,
        derive_glm_moe_geometry,
        validate_glm_nvfp4_bank_geometry,
    )

    config = SimpleNamespace(
        num_layers=6,
        first_k_dense_replace=2,
        num_experts=7,
        hidden_size=32,
        moe_intermediate_size=48,
        num_experts_per_tok=3,
    )

    assert derive_glm_moe_geometry(config) == GlmMoeGeometry(
        num_experts=7,
        hidden_size=32,
        intermediate_size=48,
        num_layers=4,
        top_k=3,
    )

    geometry = derive_glm_moe_geometry(config)
    e, h, i, layers = (
        geometry.num_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_layers,
    )
    sources = {
        "gate_up_packed": [torch.empty(e, 2 * i, h // 2) for _ in range(layers)],
        "gate_up_scale": [torch.empty(e, 2 * i, h // 16) for _ in range(layers)],
        "gate_up_global": [torch.empty(e, 2 * i) for _ in range(layers)],
        "down_packed": [torch.empty(e, h, i // 2) for _ in range(layers)],
        "down_scale": [torch.empty(e, h, i // 16) for _ in range(layers)],
        "down_global": [torch.empty(e, h) for _ in range(layers)],
    }
    assert validate_glm_nvfp4_bank_geometry(config, sources) is sources


@pytest.mark.parametrize(
    "root_cls,block_cls",
    [
        (
            "freetoken.models.glm4_moe.model.Glm4MoeForCausalLM",
            "freetoken.models.glm4_moe.moe.Glm4MoeSparseBlock",
        ),
        (
            "freetoken.models.glm_moe_dsa.model.GlmMoeDsaForCausalLM",
            "freetoken.models.glm_moe_dsa.moe.GlmMoeDsaSparseBlock",
        ),
    ],
)
def test_glm_shared_expert_is_never_registered_with_tier(root_cls, block_cls):
    from freetoken.moe.offload_cache import attach_offload_moe_cache

    def load(path):
        module_name, class_name = path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    _init_tp()
    block = load(block_cls)(_config(), layer_id=2)
    root_type = load(root_cls)

    class Harness:
        _iter_offload_moe_layers = root_type._iter_offload_moe_layers

    root = Harness()
    root.model = SimpleNamespace(
        layers=SimpleNamespace(
            op_list=[SimpleNamespace(mlp=object()), SimpleNamespace(mlp=block)]
        )
    )
    cache = SimpleNamespace()

    registered = attach_offload_moe_cache(root, cache)

    assert registered == [block.experts]
    assert block.experts.offload_cache is cache
    assert not hasattr(block.shared_experts, "offload_cache")


def test_glm_session_profile_capture_fires_from_routed_decode(monkeypatch):
    from freetoken.models.glm_moe_dsa.moe import GlmMoeDsaSparseBlock

    class Executor:
        def decode(self, layer_id, hidden, weights, ids):
            return torch.zeros_like(hidden)

    class Cache:
        decode_target = "cpu"
        cpu_executor = Executor()

        def __init__(self):
            self.profile_routes = []

        def is_gpufetch_layer(self, layer_id):
            return False

        def is_cpu_layer(self, layer_id):
            return True

        def _record_session_profile(self, layer_id, expert_ids):
            self.profile_routes.append((layer_id, expert_ids.clone()))

        def record_decode_frequency(self, layer_id, expert_ids):
            self._record_session_profile(layer_id, expert_ids)

    _init_tp()
    block = GlmMoeDsaSparseBlock(_config(), layer_id=2)
    cache = Cache()
    block.experts.offload_cache = cache
    monkeypatch.setattr(
        block.experts,
        "routed_forward",
        lambda hidden, weights, ids: block.experts._decode_routed(hidden, weights, ids),
    )
    monkeypatch.setattr(
        block.shared_experts, "forward", lambda hidden: torch.zeros_like(hidden)
    )
    with torch.no_grad():
        block.gate.weight.zero_()
        block.e_score_correction_bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    block.forward(torch.zeros(1, 16))

    assert len(cache.profile_routes) == 1
    assert cache.profile_routes[0][0] == 0
    assert cache.profile_routes[0][1].tolist() == [[0, 1]]


def test_glm_disk_hot_and_session_flags_pass_arch_neutral_gate(monkeypatch):
    from freetoken.attention import AttnType
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _adjust_config
    from freetoken.models.config import KVCacheGroupSpec

    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=(0, 1, 2, 3, 4),
        num_kv_heads=1,
        head_dim=32,
        sliding_window=None,
        mla=True,
        index_head_dim=16,
        num_index_layers=5,
        attn_type=AttnType.DSA,
    )
    model_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        single_stream_only=False,
        dsv4_args=None,
        has_swa_attention=False,
        has_linear_attention=False,
        is_moe=True,
        expert_quant="nvfp4",
        hidden_act="silu",
        moe_weight_format=None,
        num_layers=5,
        num_moe_layers=3,
        num_experts=4,
        rotary_config=SimpleNamespace(max_position=1024),
        kv_cache_group_specs=lambda: (spec,),
    )
    config = EngineConfig(
        model_path="/synthetic/glm.ftw",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="dsa",
        moe_backend="offload",
        moe_cache_size=8,
        moe_disk_layers="1.0",
        moe_disk_layer_profile="profile.json",
        moe_hot_expert_budget_gib=1.0,
        moe_hot_adapt_halflife_steps=321,
        moe_hot_adapt_interval_steps=17,
        moe_hot_adapt_max_swap_gib=0.25,
        moe_disk_prefill="cpu",
        moe_disk_decode="cpu",
        moe_disk_pager="uffd",
        moe_disk_lookahead="on",
        moe_pager_budget_gib=9.0,
        session_expert_prefetch="on",
        session_protect_experts=12,
    )
    object.__setattr__(config, "model_config", model_config)
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: True)

    _adjust_config(config)

    assert config.moe_backend == "offload"
    assert config.moe_disk_layers == "1.0"
    assert config.moe_disk_layer_profile == "profile.json"
    assert config.moe_hot_expert_budget_gib == 1.0
    assert config.moe_hot_adapt_halflife_steps == 321
    assert config.moe_hot_adapt_interval_steps == 17
    assert config.moe_hot_adapt_max_swap_gib == 0.25
    assert config.moe_disk_prefill == "cpu"
    assert config.moe_disk_decode == "cpu"
    assert config.moe_disk_pager == "uffd"
    assert config.moe_disk_lookahead == "on"
    assert config.moe_pager_budget_gib == 9.0
    assert config.session_expert_prefetch == "on"
    assert config.session_protect_experts == 12


def test_glm_ple_backend_is_rejected_without_attribute_crashes():
    from freetoken.engine.engine import _gate_ple_settings

    config = SimpleNamespace(ple_backend="disk")
    model_config = SimpleNamespace(model_type="glm_moe_dsa")

    with pytest.raises(ValueError, match="PLE n-gram table"):
        _gate_ple_settings(config, model_config, setattr)


def test_glm_auxiliary_ple_flags_are_ignored_with_clear_warning(monkeypatch):
    from freetoken.engine import engine

    warnings = []
    monkeypatch.setattr(engine.logger, "warning_rank0", warnings.append)
    config = SimpleNamespace(
        ple_backend="pinned",
        ple_prefill_gather="off",
        ple_cache_gib=3.0,
        ple_cache_warm="warm.json",
        ple_cache_profile_out="profile.json",
    )
    model_config = SimpleNamespace(model_type="glm4_moe")

    def override(name, value):
        setattr(config, name, value)

    assert not engine._gate_ple_settings(config, model_config, override)
    assert config.ple_prefill_gather == "on"
    assert config.ple_cache_gib == 8.0
    assert config.ple_cache_warm is None
    assert config.ple_cache_profile_out is None
    assert len(warnings) == 1
    assert "glm4_moe has no PLE n-gram table" in warnings[0]
    assert "--ple-cache-warm='warm.json'" in warnings[0]


def test_glm_ple_gate_accepts_legacy_minimal_stub():
    from freetoken.engine.engine import _gate_ple_settings

    assert not _gate_ple_settings(
        SimpleNamespace(), SimpleNamespace(model_type="glm_moe_dsa"), setattr
    )
