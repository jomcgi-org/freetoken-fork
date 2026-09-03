# SPDX-License-Identifier: Apache-2.0
"""Fused n-gram hash to PLE table row ids.

The eager hash builds a packed ragged window, scans boundaries, and launches many
small elementwise operations. This kernel performs the same signed int64 arithmetic
with one Triton program per token and without materializing the packed window.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_row_ids_kernel(
    ids_ptr,
    ctx_ptr,
    req_ptr,
    local_ptr,
    mult_ptr,
    vocab_ptr,
    off_ptr,
    out_ptr,
    EOS: tl.constexpr,
    CTX_LEN: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS_PER: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    req = tl.load(req_ptr + token).to(tl.int64)
    local = tl.load(local_ptr + token).to(tl.int64)

    head = tl.arange(0, BLOCK_H)
    head_ok = head < NUM_HEADS
    mixed = tl.load(ids_ptr + token).to(tl.int64) * tl.load(mult_ptr).to(tl.int64)
    acc = tl.zeros([BLOCK_H], dtype=tl.int64)

    valid = 1
    for shift in tl.static_range(1, NGRAM):
        column = CTX_LEN + local - shift
        from_ids = column >= CTX_LEN
        from_ctx = (column >= 0) & (column < CTX_LEN)
        token_ids = tl.load(ids_ptr + (token - shift), mask=from_ids, other=0)
        token_ctx = tl.load(ctx_ptr + req * CTX_LEN + column, mask=from_ctx, other=EOS)
        raw = tl.where(from_ids, token_ids, token_ctx).to(tl.int64)
        valid = valid * tl.where((column >= 0) & (raw != EOS), 1, 0)
        mixed = mixed ^ (
            tl.where(valid == 1, raw, EOS) * tl.load(mult_ptr + shift).to(tl.int64)
        )
        ngram = shift + 1
        block = (head >= (ngram - 2) * HEADS_PER) & (head < (ngram - 1) * HEADS_PER)
        acc = tl.where(block, mixed, acc)

    vocab = tl.load(vocab_ptr + head, mask=head_ok, other=1).to(tl.int64)
    offset = tl.load(off_ptr + head, mask=head_ok, other=0).to(tl.int64)
    rem = acc % vocab
    rem = tl.where(rem < 0, rem + vocab, rem)
    tl.store(out_ptr + token * NUM_HEADS + head, rem + offset, mask=head_ok)


def ple_row_ids(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    req_index: torch.Tensor,
    local_index: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Return ``[tokens, heads]`` int64 global PLE table row ids."""
    tokens = input_ids.numel()
    ngram_size = int(multipliers.numel())
    num_heads = int(vocab_sizes.numel())
    ctx_len = int(ngram_context.shape[-1])
    if ctx_len != ngram_size - 1:
        raise ValueError(
            f"PLE hash context has {ctx_len} ids but ngram_size {ngram_size} "
            f"needs {ngram_size - 1}"
        )
    if num_heads != heads_per_ngram * (ngram_size - 1):
        raise ValueError(
            f"PLE hash has {num_heads} heads, expected heads_per_ngram "
            f"{heads_per_ngram} x {ngram_size - 1} n-gram orders"
        )
    out = torch.empty(
        (tokens, num_heads), dtype=torch.int64, device=input_ids.device
    )
    if tokens == 0:
        return out
    assert ngram_context.is_contiguous()
    assert out.is_contiguous()
    _ple_row_ids_kernel[(tokens,)](
        input_ids,
        ngram_context,
        req_index,
        local_index,
        multipliers,
        vocab_sizes,
        offsets,
        out,
        EOS=int(eos_token_id),
        CTX_LEN=ctx_len,
        NGRAM=ngram_size,
        HEADS_PER=int(heads_per_ngram),
        NUM_HEADS=num_heads,
        BLOCK_H=triton.next_power_of_2(num_heads),
        num_warps=1,
    )
    return out


__all__ = ["ple_row_ids"]
