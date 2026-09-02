from __future__ import annotations

from types import SimpleNamespace

import torch


def test_geometry_accepts_alias_and_noncontiguous_sparse_layers():
    from freetoken.models.glm_moe import derive_glm_moe_geometry, glm_moe_bank_layer

    cfg = SimpleNamespace(
        num_layers=4,
        mlp_layer_types=("sparse", "dense", "sparse", "dense"),
        first_k_dense_replace=0,
        n_routed_experts=7,
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts_per_tok=3,
    )
    geometry = derive_glm_moe_geometry(cfg)
    assert (geometry.num_experts, geometry.num_layers) == (7, 2)
    assert glm_moe_bank_layer(cfg, 0) == 0
    assert glm_moe_bank_layer(cfg, 2) == 1


def test_weight_routing_skips_vision_mtp_experts_and_maps_text():
    from freetoken.models.glm5_next.weight import _rename

    assert _rename("model.visual.blocks.0.attn.qkv.weight") is None
    assert _rename("model.language_model.layers.45.self_attn.q_proj.weight") is None
    assert (
        _rename("model.language_model.layers.3.mlp.experts.2.gate_proj.weight_packed")
        is None
    )
    assert _rename("model.language_model.layers.3.self_attn.q_a_proj.weight") == (
        "model.layers.3.self_attn.q_a_proj.weight"
    )
    assert _rename("model.language_model.layers.4.self_attn.f_a_proj.weight") == (
        "model.layers.4.self_attn.f_a_proj.weight"
    )
    assert _rename(
        "model.language_model.layers.4.mlp.gate.e_score_correction_bias"
    ) == ("model.layers.4.mlp.e_score_correction_bias")
    assert _rename(
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape"
    ) == ("model.layers.3.self_attn.indexer.index_kpool_compress_ape")
    assert _rename(
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate"
    ) == ("model.layers.3.self_attn.indexer.index_kpool_compress_gate")


def test_compressed_tensors_nvfp4_source_alias_and_global_reciprocal():
    from freetoken.models.glm5_next.weight import _NVFP4_SOURCE_SPEC
    from freetoken.models.nvfp4_banks import _global_scale, _source_kind

    match = _NVFP4_SOURCE_SPEC.key_pattern.match(
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_packed"
    )
    assert match is not None and _source_kind(_NVFP4_SOURCE_SPEC, match) == "weight"
    got = _global_scale(_NVFP4_SOURCE_SPEC, torch.tensor(4.0))
    assert got.dtype == torch.float16 and got.item() == 0.25


def test_converter_model_registration():
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Glm5NextForConditionalGeneration")
    assert spec.module == "freetoken.models.glm5_next"
    assert spec.model_cls == "Glm5NextForCausalLM"
    assert spec.iter_weights == "iter_weights"
