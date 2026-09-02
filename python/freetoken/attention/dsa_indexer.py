"""GLM DSA (IndexShare lightning indexer) addressing, mixed into the backend.

Same contract as ``dsv4_indexer.py``: the indexer's weights/projections stay in the
MODEL (it hands per-token q/k/weights in); this mixin owns the ADDRESSING -- scoring
against the paged index-key slab, causal top-k selection, and the map from selected
positions to physical rows. ``-1`` is a gather-only sentinel (the sparse-attention
kernel masks it, it never reaches a store), and device-side counts bound every kernel
so CUDA-graph replays track the live position, not the staged width.

GLM-5.2 keeps the original token-granular ``ratio=1`` selection. GLM-5.3 caches
``key | compression_gate`` per physical token, rebuilds complete learned pools,
selects at ``ratio=index_kpool``, expands selected blocks to token positions, and
appends the causal incomplete tail. Both modes share DSV4's selection helpers.
"""

from __future__ import annotations

import torch

from .dsv4_indexer import IndexerBackendMixin


def dsa_pool_index_states(
    packed_states: torch.Tensor, ape: torch.Tensor, index_kpool: int
) -> torch.Tensor:
    """Build complete GLM-5.3 pooled keys from cached token state.

    ``packed_states[..., T, 2D]`` stores ``key | compression_gate``. For each
    complete group and each channel ``d`` the exact reference arithmetic is::

        p[i, d] = softmax_i(float(gate[i, d]) + float(ape[i, d]))
        pooled[d] = sum_i(cast(p[i, d], key.dtype) * key[i, d])

    The incomplete tail is not compressed. It is appended directly during block
    expansion, matching ``index_kpool_always_select_tail``.
    """
    if index_kpool <= 1:
        raise ValueError("pooled DSA needs index_kpool > 1")
    if packed_states.shape[-1] % 2:
        raise ValueError(
            "pooled DSA token state must contain equally sized key and gate"
        )
    head_dim = packed_states.shape[-1] // 2
    if ape.shape != (index_kpool, head_dim):
        raise ValueError(
            f"pooled DSA APE must be {(index_kpool, head_dim)}, got {tuple(ape.shape)}"
        )
    complete = packed_states.shape[-2] // index_kpool
    grouped = packed_states[..., : complete * index_kpool, :].reshape(
        *packed_states.shape[:-2], complete, index_kpool, 2 * head_dim
    )
    keys, gates = grouped.split(head_dim, dim=-1)
    probabilities = (gates.float() + ape.float()).softmax(dim=-2).to(keys.dtype)
    return (probabilities * keys).sum(dim=-2)


def dsa_expand_block_positions(
    block_picks: torch.Tensor,
    visible_tokens: torch.Tensor,
    *,
    index_kpool: int,
    token_topk: int,
    always_select_tail: bool = True,
) -> torch.Tensor:
    """Expand selected pooled blocks and compact the visible incomplete tail.

    The output is request-logical token positions with ``-1`` padding and fixed
    width ``token_topk + index_kpool - 1`` when the tail is enabled. ``block_picks`` and
    ``visible_tokens`` share all leading dimensions.
    """
    if token_topk % index_kpool:
        raise ValueError("DSA token top-k must be divisible by index_kpool")
    block_topk = token_topk // index_kpool
    if block_picks.shape[-1] > block_topk:
        raise ValueError(
            f"pooled DSA picks width exceeds {block_topk}: {block_picks.shape[-1]}"
        )
    if block_picks.shape[-1] < block_topk:
        padding = block_picks.new_full(
            (*block_picks.shape[:-1], block_topk - block_picks.shape[-1]), -1
        )
        block_picks = torch.cat([block_picks, padding], dim=-1)
    visible = visible_tokens.to(torch.long)
    output_width = token_topk + (index_kpool - 1 if always_select_tail else 0)
    columns = torch.arange(output_width, device=block_picks.device, dtype=torch.long)
    columns = columns.view(*([1] * visible.ndim), output_width)
    complete = torch.minimum(
        visible // index_kpool,
        visible.new_full((), block_topk),
    )
    expanded_count = complete * index_kpool
    block_rank = (columns // index_kpool).clamp(max=block_topk - 1)
    chosen = block_picks.to(torch.long).gather(
        -1, block_rank.expand(*block_picks.shape[:-1], output_width)
    )
    expanded = chosen * index_kpool + columns.remainder(index_kpool)
    tail_start = (visible // index_kpool) * index_kpool
    tail_offset = columns - expanded_count.unsqueeze(-1)
    is_expanded = columns < expanded_count.unsqueeze(-1)
    is_tail = (
        (columns >= expanded_count.unsqueeze(-1))
        & (tail_offset < visible.remainder(index_kpool).unsqueeze(-1))
        & (tail_offset < index_kpool - 1)
    )
    if not always_select_tail:
        is_tail = torch.zeros_like(is_tail)
    positions = torch.where(
        is_expanded,
        expanded,
        tail_start.unsqueeze(-1) + tail_offset,
    )
    valid = (
        (is_expanded | is_tail) & (positions >= 0) & (positions < visible.unsqueeze(-1))
    )
    return torch.where(valid, positions, positions.new_full((), -1)).to(torch.int32)


class DSAIndexerMixin(IndexerBackendMixin):
    def dsa_decode_scores(
        self,
        q_idx: torch.Tensor,
        w: torch.Tensor,
        slot: int,
        rows: torch.Tensor,
        kvlen: torch.Tensor,
    ) -> torch.Tensor:
        """Fused head-reduced logits ``[bs, W]`` fp32 for a decode step: keys gathered
        off the row snapshot inside the kernel, live length read from device memory,
        ``-inf`` past it (so the shared select's -inf ordering holds)."""
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_decode_logits

        return glm_dsa_decode_logits(
            q_idx, w * self.index_scale, self.kvcache.index_k_cache(slot), rows, kvlen
        )

    def dsa_pooled_decode_scores(
        self,
        q_idx: torch.Tensor,
        w: torch.Tensor,
        ape: torch.Tensor,
        slot: int,
        rows: torch.Tensor,
        kvlen: torch.Tensor,
    ) -> torch.Tensor:
        """Fused GLM-5.3 complete-pool logits for one decode step."""
        from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_pooled_decode_logits

        return glm_dsa_pooled_decode_logits(
            q_idx,
            w * self.index_scale,
            self.kvcache.index_k_cache(slot),
            ape,
            rows,
            kvlen,
            self.index_kpool,
        )

    def dsa_prefill_logits(
        self, q_idx: torch.Tensor, k_all: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Head-reduced logits ``[m, kv_len]`` fp32 over a dense key slab (dsv4's
        fused ``indexer_logits`` -- no per-head transient)."""
        from freetoken.kernel.triton.dsv4.indexer import indexer_logits

        return indexer_logits(
            q_idx.unsqueeze(0),
            k_all.unsqueeze(0),
            (w * self.index_scale).to(torch.float32).unsqueeze(0),
        )[0]

    def dsa_pooled_prefill_logits(
        self,
        q_idx: torch.Tensor,
        packed_states: torch.Tensor,
        w: torch.Tensor,
        ape: torch.Tensor,
    ) -> torch.Tensor:
        pooled = dsa_pool_index_states(packed_states, ape, self.index_kpool)
        return self.dsa_prefill_logits(q_idx, pooled, w)

    @staticmethod
    def dsa_map_rows(picks: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Selected POSITIONS -> physical rows, ``-1`` sentinel passed through.

        ``picks`` [..., K] int (from the shared select fns), ``rows`` broadcastable
        position-ordered physical rows. Returns int32."""
        sel = rows.gather(-1, picks.clamp_min(0).long()).to(torch.int32)
        return torch.where(picks < 0, sel.new_full((), -1), sel)


__all__ = [
    "DSAIndexerMixin",
    "dsa_expand_block_positions",
    "dsa_pool_index_states",
]
