from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
)

from .common import fill, parsed_config


def test_wrapper_unwrap_and_hybrid_groups():
    cfg = parsed_config()
    assert cfg.model_type == "glm5_next"
    assert cfg.architectures == ["Glm5NextForConditionalGeneration"]
    assert cfg.vision_config is None and cfg.image_token_id == 29
    assert cfg.num_layers == 4
    assert cfg.num_moe_layers == 1
    assert cfg.expert_quant == "nvfp4"
    assert cfg.glm5_next_args.num_nextn_predict_layers == 1
    linear = next(
        g for g in cfg.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig)
    )
    mla = next(
        g for g in cfg.attention_groups if isinstance(g, FullAttentionGroupConfig)
    )
    assert linear.layer_ids == (0, 1, 2)
    assert (linear.num_key_heads, linear.key_head_dim, linear.conv_kernel_dim) == (
        2,
        4,
        3,
    )
    assert mla.layer_ids == (3,) and mla.mla
    assert mla.head_dim == 4 and mla.num_index_layers == 1


def test_per_layer_attention_and_mlp_composition():
    from freetoken.models.glm5_next.linear import Glm5NextKimiDeltaAttention
    from freetoken.models.glm5_next.mlp import Glm5NextGatedMLP
    from freetoken.models.glm5_next.model import Glm5NextDecoderLayer
    from freetoken.models.glm5_next.moe import Glm5NextSparseBlock
    from freetoken.models.glm_moe_dsa.attention import GlmMoeDsaAttention

    cfg = parsed_config()
    layers = [Glm5NextDecoderLayer(cfg, i) for i in range(cfg.num_layers)]
    assert all(
        isinstance(layer.self_attn, Glm5NextKimiDeltaAttention) for layer in layers[:3]
    )
    assert isinstance(layers[3].self_attn, GlmMoeDsaAttention)
    assert all(isinstance(layer.mlp, Glm5NextGatedMLP) for layer in layers[:3])
    assert isinstance(layers[3].mlp, Glm5NextSparseBlock)
    assert layers[3].mlp.experts.layer_id == 0
    assert layers[3].mlp.experts.activation == "clamped_silu"


def test_linear_reference_and_mla_fake_backend_shapes(monkeypatch):
    import freetoken.core as core
    from freetoken.core import Context
    from freetoken.models.glm5_next.linear import Glm5NextKimiDeltaAttention
    from freetoken.models.glm_moe_dsa.attention import GlmMoeDsaAttention

    cfg = parsed_config()
    linear = Glm5NextKimiDeltaAttention(cfg, 0)
    fill(linear)
    x = torch.randn(1, 5, cfg.hidden_size)
    assert linear.forward_reference(x).shape == x.shape
    g, beta = linear._gate_params(x.reshape(-1, cfg.hidden_size))
    assert g.shape == (5, 2, 4) and beta.shape == (5, 2)
    assert torch.all(g <= 0) and torch.all(g >= -5)

    class Backend:
        dsa_enabled = False

        def mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, **kwargs):
            assert layer_id == 3
            assert q_pe.shape[-1] == k_rope.shape[-1] == 0
            return q_nope.new_zeros(q_nope.shape)

    ctx = Context(page_size=1)
    ctx.attn_backend = Backend()
    core._GLOBAL_CTX = ctx
    batch = SimpleNamespace(positions=torch.arange(5))
    attention = GlmMoeDsaAttention(cfg, 3)
    fill(attention, seed=2)
    try:
        with ctx.forward_batch(batch):
            assert attention.forward(torch.randn(5, cfg.hidden_size)).shape == (
                5,
                cfg.hidden_size,
            )
    finally:
        core._GLOBAL_CTX = None


def test_compact_mla_pool_uses_global_layer_ids():
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.dsa_pool import DSAKVCache

    cfg = parsed_config()
    pool = create_kvcache_pool(
        cfg,
        num_pages=2,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert isinstance(pool, DSAKVCache)
    assert pool.num_layers == 1
    pool.store_kv(torch.ones(1, 4), torch.empty(1, 0), torch.tensor([0]), layer_id=3)
    assert torch.equal(pool.latent_rows(3)[0], torch.ones(4))
