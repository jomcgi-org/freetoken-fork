# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA and HMM row gathers for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in host
memory and the GPU dereferences it in place over PCIe. The pinned backend uses its host VA
on Linux/UVA and mapped device address on WDDM (``kernel/pinned.device_ptr``). The HMM
backend uses a device table of read-only file-mapping addresses. One program per requested
row reads the row, widens to fp32, applies the per-tensor scale, and stores bf16.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    shard_bases_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    rows_per_shard,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    IDS_RAW_INT32: tl.constexpr,
    SHARDED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if IDS_RAW_INT32:
        ids_base = ids_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
        idx = tl.load(ids_base + row).to(tl.int64)
    else:
        idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    if SHARDED:
        shard = idx // rows_per_shard
        idx = idx % rows_per_shard
        table_address = tl.load(shard_bases_ptr + shard)
    else:
        table_address = table_ptr
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if IS_FP8:
        if e4m3_native_cx():
            base = table_address.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_address.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0))
    else:
        base = table_address.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.where(in_range, values * scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            0,
            row_ids,
            out,
            float(scale),
            num_rows,
            0,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            IDS_RAW_INT32=False,
            SHARDED=False,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


def ple_gather_rows_from_ptr(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids_ptr: int,
    num_ids: int,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather int32 ids from a fixed mapped-host address.

    This is the CUDA-graph disk-staging path. Both the staged table and compact ids live in
    pinned host allocations whose addresses remain stable across replays.
    """
    assert out.shape == (num_ids, embed_dim) and out.is_contiguous(), out.shape
    if num_ids:
        _ple_gather_kernel[(num_ids,)](
            table_ptr,
            0,
            row_ids_ptr,
            out,
            float(scale),
            num_rows,
            0,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            IDS_RAW_INT32=True,
            SHARDED=False,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


def ple_gather_rows_sharded(
    shard_bases: torch.Tensor,
    rows_per_shard: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather device row ids directly from file-backed HMM shard mappings.

    ``shard_bases`` is a device int64 tensor containing the host virtual address of
    each read-only mapping. HMM resolves GPU faults on those mappings without CUDA
    host registration or an intermediate pinned bank.
    """
    n = row_ids.numel()
    assert shard_bases.dtype is torch.int64 and shard_bases.is_cuda
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            0,
            shard_bases,
            row_ids,
            out,
            float(scale),
            num_rows,
            rows_per_shard,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            IDS_RAW_INT32=False,
            SHARDED=True,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_rows", "ple_gather_rows_from_ptr", "ple_gather_rows_sharded"]
