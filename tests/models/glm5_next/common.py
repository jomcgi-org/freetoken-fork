from __future__ import annotations

from types import SimpleNamespace

from freetoken.distributed import set_tp_info, try_get_tp_info


def init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def hf_config():
    text = SimpleNamespace(
        model_type="glm5_next_text",
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        first_k_dense_replace=3,
        hidden_size=8,
        vocab_size=32,
        intermediate_size=12,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        rope_interleave=True,
        indexer_rope_interleave=True,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=8,
        indexer_types=["full"] * 4,
        linear_num_heads=2,
        linear_head_dim=4,
        linear_conv_kernel_dim=3,
        linear_lower_bound=-5.0,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
            "gate_lower_bound": -5.0,
        },
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        mhc=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=4,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        swiglu_limit=10.0,
        num_nextn_predict_layers=1,
    )
    return SimpleNamespace(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
        text_config=text,
        vision_config=SimpleNamespace(model_type="glm5_next_vision"),
        image_token_id=29,
        video_token_id=30,
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {
                "experts": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                    }
                }
            },
        },
    )


def parsed_config():
    from freetoken.models.glm5_next.config import parse_config

    init_tp()
    config = parse_config(hf_config())
    object.__setattr__(config, "moe_backend", "offload")
    return config


def fill(op, seed: int = 1):
    import torch

    generator = torch.Generator().manual_seed(seed)
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05, generator=generator)
        else:
            tensor.zero_()
