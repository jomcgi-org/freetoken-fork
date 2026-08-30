# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA and HMM row gathers for the Qwen3.8-Flash-Next PLE n-gram table.

The table stays in host memory and the GPU dereferences it in place over PCIe. It may use
FP8 with scalar or per-row scales, INT4 group-16, or e2m1 group-16. The pinned backend uses
its host VA on Linux/UVA and mapped device address on WDDM (``kernel/pinned.device_ptr``).
The HMM backend uses device tables of read-only file-mapping addresses. One program per
requested row reads and dequantizes the row, then stores bf16.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1

_FORMAT_BF16 = 0
_FORMAT_FP8 = 1
_FORMAT_FP8_ROW = 2
_FORMAT_INT4G16 = 3
_FORMAT_E2M1G16 = 4


@triton.jit
def _e2m1_to_f32(code):
    """Decode e2m1 with the same fp16 bit pattern used by the NVFP4 linear kernels."""
    bits = ((code & 7) << 9) | ((code & 8) << 12)
    value = bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return value * 16384.0


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    shard_bases_ptr,
    scale_ptr,
    scale_shard_bases_ptr,
    ids_ptr,
    out_ptr,
    global_scale,
    num_rows,
    rows_per_shard,
    EMB_DIM: tl.constexpr,
    TABLE_FORMAT: tl.constexpr,
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
        if TABLE_FORMAT == 2 or TABLE_FORMAT >= 3:
            scale_address = tl.load(scale_shard_bases_ptr + shard)
    else:
        table_address = table_ptr
        scale_address = scale_ptr
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if TABLE_FORMAT == 1 or TABLE_FORMAT == 2:
        if e4m3_native_cx():
            base = table_address.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_address.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0))
        if TABLE_FORMAT == 2:
            scales = scale_address.to(tl.int64).to(tl.pointer_type(tl.float32))
            row_scale = tl.load(scales + idx).to(tl.float32)
        else:
            row_scale = global_scale
        values *= row_scale
    elif TABLE_FORMAT == 3 or TABLE_FORMAT == 4:
        packed_dim: tl.constexpr = EMB_DIM // 2
        groups_per_row: tl.constexpr = EMB_DIM // 16
        base = table_address.to(tl.int64).to(tl.pointer_type(tl.uint8))
        packed = tl.load(
            base + idx * packed_dim + offsets // 2, mask=mask, other=0
        ).to(tl.int32)
        codes = tl.where((offsets & 1) == 0, packed & 0xF, (packed >> 4) & 0xF)
        scale_index = idx * groups_per_row + offsets // 16
        if TABLE_FORMAT == 3:
            scales = scale_address.to(tl.int64).to(tl.pointer_type(tl.float16))
            group_scale = tl.load(scales + scale_index, mask=mask, other=0.0).to(tl.float32)
            values = (codes.to(tl.float32) - 8.0) * group_scale
        else:
            # The loader folds each shard's FP32 weight_scale_2 into its native FP8
            # group scales and widens the serving bank to FP16.
            scales = scale_address.to(tl.int64).to(tl.pointer_type(tl.float16))
            group_scale = tl.load(
                scales + scale_index, mask=mask, other=0.0
            ).to(tl.float32)
            values = _e2m1_to_f32(codes) * group_scale * global_scale
    else:
        base = table_address.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        values *= global_scale
    values = tl.where(in_range, values, 0.0)
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
    *,
    table_format: str | None = None,
    scale_ptr: int = 0,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    format_code = _format_code(table_format, is_fp8, scale_ptr)
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            0,
            scale_ptr,
            0,
            row_ids,
            out,
            float(scale),
            num_rows,
            0,
            EMB_DIM=embed_dim,
            TABLE_FORMAT=format_code,
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
    *,
    table_format: str | None = None,
    scale_ptr: int = 0,
) -> torch.Tensor:
    """Gather int32 ids from a fixed mapped-host address.

    This is the CUDA-graph disk-staging path. Both the staged table and compact ids live in
    pinned host allocations whose addresses remain stable across replays.
    """
    assert out.shape == (num_ids, embed_dim) and out.is_contiguous(), out.shape
    format_code = _format_code(table_format, is_fp8, scale_ptr)
    if num_ids:
        _ple_gather_kernel[(num_ids,)](
            table_ptr,
            0,
            scale_ptr,
            0,
            row_ids_ptr,
            out,
            float(scale),
            num_rows,
            0,
            EMB_DIM=embed_dim,
            TABLE_FORMAT=format_code,
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
    *,
    table_format: str | None = None,
    scale_shard_bases: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather device row ids directly from file-backed HMM shard mappings.

    ``shard_bases`` is a device int64 tensor containing the host virtual address of
    each read-only mapping. HMM resolves GPU faults on those mappings without CUDA
    host registration or an intermediate pinned bank.
    """
    n = row_ids.numel()
    assert shard_bases.dtype is torch.int64 and shard_bases.is_cuda
    format_code = _format_code(
        table_format, is_fp8, 0 if scale_shard_bases is None else 1
    )
    if format_code in (_FORMAT_FP8_ROW, _FORMAT_INT4G16, _FORMAT_E2M1G16):
        assert scale_shard_bases is not None
        assert scale_shard_bases.dtype is torch.int64 and scale_shard_bases.is_cuda
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            0,
            shard_bases,
            0,
            0 if scale_shard_bases is None else scale_shard_bases,
            row_ids,
            out,
            float(scale),
            num_rows,
            rows_per_shard,
            EMB_DIM=embed_dim,
            TABLE_FORMAT=format_code,
            IDS_RAW_INT32=False,
            SHARDED=True,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


def ple_gather_rows_sharded_from_ptr(
    shard_bases: torch.Tensor,
    rows_per_shard: int,
    num_rows: int,
    embed_dim: int,
    row_ids_ptr: int,
    num_ids: int,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
    *,
    table_format: str | None = None,
    scale_shard_bases: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather fixed-address int32 slot ids from pinned host slabs."""
    assert shard_bases.dtype is torch.int64 and shard_bases.is_cuda
    format_code = _format_code(
        table_format, is_fp8, 0 if scale_shard_bases is None else 1
    )
    if format_code in (_FORMAT_FP8_ROW, _FORMAT_INT4G16, _FORMAT_E2M1G16):
        assert scale_shard_bases is not None
        assert scale_shard_bases.dtype is torch.int64 and scale_shard_bases.is_cuda
    assert out.shape == (num_ids, embed_dim) and out.is_contiguous(), out.shape
    if num_ids:
        _ple_gather_kernel[(num_ids,)](
            0,
            shard_bases,
            0,
            0 if scale_shard_bases is None else scale_shard_bases,
            row_ids_ptr,
            out,
            float(scale),
            num_rows,
            rows_per_shard,
            EMB_DIM=embed_dim,
            TABLE_FORMAT=format_code,
            IDS_RAW_INT32=True,
            SHARDED=True,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


def _format_code(table_format: str | None, is_fp8: bool, scale_ptr: int) -> int:
    if table_format is None:
        return _FORMAT_FP8 if is_fp8 else _FORMAT_BF16
    if table_format == "fp8":
        return _FORMAT_FP8_ROW if scale_ptr else _FORMAT_FP8
    if table_format == "int4g16":
        if not scale_ptr:
            raise ValueError("INT4 group-16 PLE gather requires scale storage")
        return _FORMAT_INT4G16
    if table_format == "e2m1g16":
        if not scale_ptr:
            raise ValueError("e2m1 group-16 PLE gather requires scale storage")
        return _FORMAT_E2M1G16
    raise ValueError(f"unsupported PLE table format {table_format!r}")


__all__ = [
    "ple_gather_rows",
    "ple_gather_rows_from_ptr",
    "ple_gather_rows_sharded",
    "ple_gather_rows_sharded_from_ptr",
]
