"""Shared routed-expert geometry and offload attachment for GLM MoE families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class GlmMoeGeometry:
    """The routed expert bank geometry read from a parsed GLM model config."""

    num_experts: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    top_k: int


def derive_glm_moe_geometry(config) -> GlmMoeGeometry:
    """Derive and validate the common GLM routed-expert geometry.

    GLM-4 and GLM-MoE-DSA differ in attention, not in their MoE bank layout.
    Optional attributes use ``getattr`` because lightweight test and conversion
    config objects can predate the corresponding ``ModelConfig`` convenience
    properties.
    """
    first_moe_layer = int(getattr(config, "first_k_dense_replace", 0))
    mlp_types = tuple(getattr(config, "mlp_layer_types", ()) or ())
    if mlp_types and len(mlp_types) < int(config.num_layers):
        raise ValueError(
            f"GLM mlp_layer_types has {len(mlp_types)} entries for {config.num_layers} layers"
        )
    num_layers = getattr(config, "num_moe_layers", None)
    if num_layers is None:
        num_layers = (
            sum(t == "sparse" for t in mlp_types[: int(config.num_layers)])
            if mlp_types
            else int(config.num_layers) - first_moe_layer
        )
    num_experts = getattr(config, "num_experts", None)
    if not num_experts:
        num_experts = getattr(config, "n_routed_experts", None)
    if not num_experts:
        raise ValueError("GLM config needs num_experts or n_routed_experts")
    geometry = GlmMoeGeometry(
        num_experts=int(num_experts),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.moe_intermediate_size),
        num_layers=int(num_layers),
        top_k=int(config.num_experts_per_tok),
    )
    expected_layers = (
        sum(t == "sparse" for t in mlp_types[: int(config.num_layers)])
        if mlp_types
        else int(config.num_layers) - first_moe_layer
    )
    if geometry.num_layers != expected_layers:
        raise ValueError(f"GLM num_moe_layers must equal sparse layer count {expected_layers}")
    if min(
        geometry.num_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_layers,
        geometry.top_k,
    ) <= 0:
        raise ValueError(f"GLM MoE geometry must be positive, got {geometry}")
    if geometry.top_k > geometry.num_experts:
        raise ValueError(
            f"GLM top_k {geometry.top_k} exceeds routed experts {geometry.num_experts}"
        )
    return geometry


def glm_moe_bank_layer(config, layer_id: int) -> int:
    """Packed expert-bank row for a decoder layer, including non-contiguous MLP types."""
    mlp_types = tuple(getattr(config, "mlp_layer_types", ()) or ())
    if mlp_types:
        if layer_id >= len(mlp_types) or mlp_types[layer_id] != "sparse":
            raise ValueError(f"GLM decoder layer {layer_id} is not sparse")
        return sum(t == "sparse" for t in mlp_types[:layer_id])
    first = int(getattr(config, "first_k_dense_replace", 0))
    if layer_id < first:
        raise ValueError(f"GLM decoder layer {layer_id} is dense")
    return layer_id - first


def glm_shared_expert_count(config) -> int:
    """Return the always-resident shared expert count and enforce its contract."""
    count = max(1, int(getattr(config, "n_shared_experts", 1)))
    assert count > 0, "GLM sparse MoE requires at least one always-resident shared expert"
    return count


def validate_glm_nvfp4_bank_geometry(config, sources):
    """Assert that a GLM NVFP4 bank contains routed experts only."""
    geometry = derive_glm_moe_geometry(config)
    e = geometry.num_experts
    h = geometry.hidden_size
    i = geometry.intermediate_size
    expected = {
        "gate_up_packed": (e, 2 * i, h // 2),
        "gate_up_scale": (e, 2 * i, h // 16),
        "gate_up_global": (e, 2 * i),
        "down_packed": (e, h, i // 2),
        "down_scale": (e, h, i // 16),
        "down_global": (e, h),
    }
    assert set(sources) == set(expected), (
        f"GLM NVFP4 bank names {sorted(sources)} do not match {sorted(expected)}"
    )
    for name, shape in expected.items():
        layers = sources[name]
        assert len(layers) == geometry.num_layers, (
            name,
            len(layers),
            geometry.num_layers,
        )
        for layer_id, tensor in enumerate(layers):
            assert tuple(tensor.shape) == shape, (name, layer_id, tensor.shape, shape)
    return sources


def iter_glm_offload_moe_layers(model) -> Iterator:
    """Yield only routed GLM experts for offload cache attachment.

    The shared MLP is dense and always active. This hook deliberately keeps it
    outside cache registration and asserts that no future refactor turns it into
    an ``OffloadMoELayer`` that could enter the bank or disk pager.
    """
    from freetoken.layers import OffloadMoELayer

    decoder = getattr(model, "model", model)
    layers = getattr(getattr(decoder, "layers", None), "op_list", ())
    next_layer_id = 0
    for layer in layers:
        block = getattr(layer, "mlp", None)
        routed = getattr(block, "experts", None)
        if routed is None:
            continue
        shared = getattr(block, "shared_experts", None)
        assert shared is not None, "GLM sparse MoE block lost its resident shared expert"
        assert not isinstance(shared, OffloadMoELayer), (
            "GLM shared experts are always active dense weights and must never enter "
            "the routed expert bank"
        )
        if isinstance(routed, OffloadMoELayer):
            assert routed.layer_id == next_layer_id, (
                f"GLM offload layer ids must be packed, got {routed.layer_id} "
                f"at position {next_layer_id}"
            )
            yield routed
            next_layer_id += 1


__all__ = [
    "GlmMoeGeometry",
    "derive_glm_moe_geometry",
    "glm_shared_expert_count",
    "glm_moe_bank_layer",
    "iter_glm_offload_moe_layers",
    "validate_glm_nvfp4_bank_geometry",
]
