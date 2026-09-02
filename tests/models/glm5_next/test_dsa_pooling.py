from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_pooled_key_arithmetic_matches_independent_torch_reference():
    from freetoken.attention.dsa_indexer import dsa_pool_index_states

    generator = torch.Generator().manual_seed(7)
    key = torch.randn(10, 6, generator=generator, dtype=torch.bfloat16)
    gate = torch.randn(10, 6, generator=generator, dtype=torch.bfloat16)
    ape = torch.randn(4, 6, generator=generator, dtype=torch.bfloat16)
    got = dsa_pool_index_states(torch.cat([key, gate], dim=-1), ape, 4)

    expected = []
    for block in range(2):
        start = block * 4
        block_key = key[start : start + 4]
        logits = gate[start : start + 4].float() + ape.float()
        probability = logits.softmax(dim=0).to(block_key.dtype)
        expected.append((probability * block_key).sum(dim=0))
    assert torch.equal(got, torch.stack(expected))


def test_pooled_block_expansion_maps_to_physical_rows_and_appends_tail():
    from freetoken.attention.dsa_indexer import (
        DSAIndexerMixin,
        dsa_expand_block_positions,
    )

    picks = torch.tensor([[1, 0]], dtype=torch.int64)
    logical = dsa_expand_block_positions(
        picks,
        torch.tensor([11]),
        index_kpool=4,
        token_topk=8,
    )
    assert logical.tolist() == [[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10]]

    rows = torch.tensor([[31, 7, 44, 2, 19, 80, 6, 13, 55, 9, 10]])
    physical = DSAIndexerMixin.dsa_map_rows(logical, rows)
    assert physical.tolist() == [[19, 80, 6, 13, 31, 7, 44, 2, 55, 9, 10]]


def test_config_gates_pooling_only_above_one():
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.models.glm_moe_dsa.attention import GlmDsaIndexer

    from .common import hf_config, init_tp

    init_tp()
    token_cfg = parse_config(hf_config())
    token_indexer = GlmDsaIndexer(token_cfg, 3)
    assert token_cfg.glm_dsa_args.index_kpool == 1
    assert not hasattr(token_indexer, "index_kpool_compress_ape")
    assert (
        len(
            token_indexer.compute(
                torch.randn(2, token_cfg.hidden_size),
                torch.randn(2, token_cfg.glm_dsa_args.q_lora_rank),
                torch.arange(2),
            )
        )
        == 3
    )

    pooled_hf = hf_config()
    pooled_hf.text_config.index_kpool = 4
    pooled_hf.text_config.index_kpool_always_select_tail = True
    pooled_cfg = parse_config(pooled_hf)
    pooled_indexer = GlmDsaIndexer(pooled_cfg, 3)
    assert pooled_cfg.glm_dsa_args.index_kpool == 4
    assert pooled_indexer.index_kpool_compress_ape.shape == (4, 4)
    assert pooled_indexer.index_kpool_compress_gate.shape == (4, 8)
    assert {
        "index_kpool_compress_ape",
        "index_kpool_compress_gate",
    }.issubset(pooled_indexer.state_dict())
    assert (
        len(
            pooled_indexer.compute(
                torch.randn(2, pooled_cfg.hidden_size),
                torch.randn(2, pooled_cfg.glm_dsa_args.q_lora_rank),
                torch.arange(2),
            )
        )
        == 4
    )


def test_pooled_index_state_shares_snapshot_backing_with_mla_latent():
    from freetoken.kvcache.dsa_pool import DSAKVCache

    pool = DSAKVCache(
        latent_dim=5,
        num_layers=1,
        num_pages=12,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        index_head_dim=3,
        num_index_layers=1,
        index_kpool=4,
    )
    locations = torch.tensor([2, 5, 9])
    latent = torch.randn(3, 5, dtype=torch.bfloat16)
    state = torch.randn(3, 6, dtype=torch.bfloat16)
    pool.store_kv(latent, None, locations, layer_id=0)
    pool.store_index_k(state, locations, slot=0)

    # Disk prefix capture and lazy restore operate on this same physical backing.
    snapshot = pool._kv_buffer.clone()
    pool._kv_buffer.zero_()
    pool._kv_buffer.copy_(snapshot)
    assert torch.equal(pool.latent_rows(0).index_select(0, locations), latent)
    assert torch.equal(pool.index_k_cache(0).index_select(0, locations), state)
    assert pool.unit_bytes()[0] == 5 * 2 + 6 * 2


def test_pooled_index_forces_remaining_lazy_pages_before_scoring():
    from freetoken.attention.dsa import DSAAttnBackend

    class Tracker:
        complete = False
        physical_pages = (4, 8, 12)

        def __init__(self):
            self.requested = None

        def ensure_blocks(self, blocks):
            self.requested = tuple(blocks)

    tracker = Tracker()
    batch = SimpleNamespace(reqs=[SimpleNamespace(lazy_kv_restore=tracker)])
    DSAAttnBackend._ensure_pooled_index_restored(batch)
    assert tracker.requested == (0, 1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_pooled_topk_matches_transformers_reference():
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from freetoken.attention.dsa_indexer import dsa_expand_block_positions
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_pooled_decode_logits

    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=16,
        index_head_dim=16,
        qk_rope_head_dim=0,
        index_topk=8,
        q_lora_rank=16,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )
    indexer = modeling.Glm5NextTextIndexer(config, layer_idx=0).cuda().bfloat16()
    generator = torch.Generator(device="cuda").manual_seed(29)
    with torch.no_grad():
        for parameter in indexer.parameters():
            parameter.normal_(0.0, 0.2, generator=generator)

    length = 21
    hidden = torch.randn(
        1,
        length,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q_resid = torch.randn(
        1,
        length,
        config.q_lora_rank,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    reference = indexer(
        hidden,
        q_resid,
        torch.ones(1, length, dtype=torch.bool, device="cuda"),
        None,
    )[0, -1]

    q = indexer.wq_b(q_resid).view(
        1, length, config.index_n_heads, config.index_head_dim
    )
    key = indexer.k_norm(indexer.wk(hidden)).squeeze(0)
    gate = torch.nn.functional.linear(
        hidden, indexer.index_kpool_compress_gate
    ).squeeze(0)
    weights = indexer.weights_proj(hidden).float() * (config.index_n_heads**-0.5)
    scores = glm_dsa_pooled_decode_logits(
        q[:, -1],
        weights[:, -1] * (config.index_head_dim**-0.5),
        torch.cat([key, gate], dim=-1),
        indexer.index_kpool_compress_ape,
        torch.arange(length, device="cuda", dtype=torch.int32).view(1, -1),
        torch.tensor([length], device="cuda", dtype=torch.int32),
        config.index_kpool,
    )
    blocks = scores.topk(config.index_topk // config.index_kpool, dim=-1).indices
    ours = dsa_expand_block_positions(
        blocks,
        torch.tensor([length], device="cuda"),
        index_kpool=config.index_kpool,
        token_topk=config.index_topk,
    )[0]

    reference = reference[reference >= 0].sort().values
    ours = ours[ours >= 0].sort().values
    assert torch.equal(ours, reference)
