"""Text-only config parser for the multimodal ``glm5_next`` wrapper."""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_nvfp4,
)
from freetoken.models.glm_moe_dsa.args import load_args as load_dsa_args

from .args import load_args


def parse_config(hf_config: Any) -> ModelConfig:
    """Unwrap ``text_config`` and leave all image/video ids as inert metadata."""
    text = getattr(hf_config, "text_config", hf_config)
    args = load_args(text)
    dsa = load_dsa_args(text)
    n = int(text.num_hidden_layers)
    linear_ids = tuple(
        i for i, kind in enumerate(args.layer_types) if kind == "linear_attention"
    )
    mla_ids = tuple(i for i, kind in enumerate(args.layer_types) if kind == "deepseek")
    if not linear_ids or not mla_ids:
        raise ValueError("glm5_next requires both linear_attention and deepseek layers")

    rotary = RotaryConfig(
        head_dim=dsa.qk_head_dim,
        rotary_dim=dsa.qk_rope_head_dim,
        max_position=dsa.max_position,
        base=dsa.rope_theta,
        scaling=None,
    )
    full = FullAttentionGroupConfig(
        name="mla",
        layer_ids=mla_ids,
        num_kv_heads=1,
        head_dim=dsa.kv_lora_rank + dsa.qk_rope_head_dim,
        rotary_config=rotary,
        mla=True,
        index_head_dim=dsa.index_head_dim,
        num_index_layers=sum(
            bool(dsa.indexer_types) and dsa.indexer_types[i] == "full" for i in mla_ids
        ),
    )
    linear = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=args.linear_num_heads,
        num_value_heads=args.linear_num_heads,
        key_head_dim=args.linear_head_dim,
        value_head_dim=args.linear_head_dim,
        conv_kernel_dim=args.linear_conv_kernel_dim,
        output_gate="sigmoid",
    )
    num_experts = int(
        getattr(text, "n_routed_experts", 0) or getattr(text, "num_experts", 0)
    )
    expert_quant = "nvfp4" if detect_compressed_tensors_nvfp4(hf_config) else "none"
    return ModelConfig(
        num_layers=n,
        num_qo_heads=dsa.num_heads,
        num_kv_heads=1,
        head_dim=dsa.kv_lora_rank + dsa.qk_rope_head_dim,
        hidden_size=int(text.hidden_size),
        vocab_size=int(text.vocab_size),
        intermediate_size=int(text.intermediate_size),
        hidden_act=str(text.hidden_act),
        rms_norm_eps=float(text.rms_norm_eps),
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.moe_intermediate_size),
        norm_topk_prob=bool(text.norm_topk_prob),
        model_type=getattr(hf_config, "model_type", "glm5_next"),
        architectures=getattr(
            hf_config, "architectures", ["Glm5NextForConditionalGeneration"]
        ),
        moe_enabled=True,
        expert_quant=expert_quant,
        first_k_dense_replace=int(getattr(text, "first_k_dense_replace", 0)),
        n_shared_experts=int(getattr(text, "n_shared_experts", 1)),
        routed_scaling_factor=float(getattr(text, "routed_scaling_factor", 1.0)),
        n_group=int(getattr(text, "n_group", 1)),
        topk_group=int(getattr(text, "topk_group", 1)),
        attn_sm_scale=dsa.qk_head_dim**-0.5,
        attention_groups=(linear, full),
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        swiglu_limit=args.swiglu_limit,
        moe_activation="clamped_silu",
        glm_dsa_args=dsa,
        glm5_next_args=args,
        mlp_layer_types=args.mlp_layer_types,
    )


__all__ = ["parse_config"]
