"""Model-only geometry for the GLM-5.3-Flash hybrid text decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Glm5NextArgs:
    hidden_size: int
    layer_types: Tuple[str, ...]
    mlp_layer_types: Tuple[str, ...]
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float | None
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    swiglu_limit: float
    num_nextn_predict_layers: int


def load_args(text: Any) -> Glm5NextArgs:
    linear = getattr(text, "linear_attn_config", None) or {}
    get = (
        linear.get
        if isinstance(linear, dict)
        else lambda k, d=None: getattr(linear, k, d)
    )
    n = int(text.num_hidden_layers)
    layer_types = tuple(getattr(text, "layer_types", ()) or ())
    mlp_types = tuple(getattr(text, "mlp_layer_types", ()) or ())
    if len(layer_types) != n:
        raise ValueError(
            f"glm5_next layer_types has {len(layer_types)} entries for {n} layers"
        )
    if len(mlp_types) != n:
        raise ValueError(
            f"glm5_next mlp_layer_types has {len(mlp_types)} entries for {n} layers"
        )
    layer_types = tuple(
        "deepseek" if t in ("deepseek", "deepseek_sparse_attention") else t
        for t in layer_types
    )
    unknown_attn = set(layer_types) - {"linear_attention", "deepseek"}
    unknown_mlp = set(mlp_types) - {"dense", "sparse"}
    if unknown_attn or unknown_mlp:
        raise ValueError(
            f"unknown glm5_next layer types attention={unknown_attn}, mlp={unknown_mlp}"
        )
    lower = get("gate_lower_bound", getattr(text, "linear_lower_bound", None))
    return Glm5NextArgs(
        hidden_size=int(text.hidden_size),
        layer_types=layer_types,
        mlp_layer_types=mlp_types,
        linear_num_heads=int(get("num_heads", getattr(text, "linear_num_heads", 0))),
        linear_head_dim=int(get("head_dim", getattr(text, "linear_head_dim", 0))),
        linear_conv_kernel_dim=int(
            get(
                "short_conv_kernel_size",
                getattr(text, "linear_conv_kernel_dim", 0),
            )
        ),
        linear_lower_bound=None if lower is None else float(lower),
        hc_mult=int(getattr(text, "hc_mult", 1)),
        hc_sinkhorn_iters=int(getattr(text, "hc_sinkhorn_iters", 20)),
        hc_eps=float(getattr(text, "hc_eps", 1e-6)),
        swiglu_limit=float(getattr(text, "swiglu_limit", 10.0)),
        num_nextn_predict_layers=int(getattr(text, "num_nextn_predict_layers", 0)),
    )


__all__ = ["Glm5NextArgs", "load_args"]
