"""FP8 (E4M3) per-output-row W8A16 dense linear for the non-expert projections.

At decode (bs=1..a few rows) every dense projection is a GEMV that streams its whole weight
from HBM, so on a 4090 the BF16 projections run at 800-955 GB/s and the only lever left is
bytes. This module stores those weights as ``float8_e4m3fn`` with one fp32 scale per output
row (``scale = absmax(row) / 448``; ``w ~= weight_fp8 * weight_scale[:, None]``) and reads them
directly in a W8A16 kernel: BF16 activation, FP32 accumulate, BF16 out.

Unlike ``fp8_pertensor_linear`` (checkpoint-native per-tensor scales, W8A8 where the checkpoint
carries an ``input_scale``) this path is for BF16 checkpoints quantized at load, has no
activation scale, and keeps one scheme for every M so a reply never depends on batch size.

Decode / small M (``M <= GEMV_MAX_M``): one Triton kernel, ``tl.dot`` over a
``[BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N]`` tile. The fp8 weight tile is loaded straight
along K (contiguous, 16 B per thread vectors), converted fp8 -> bf16 in registers (exact) and fed
to the tensor cores, so the activation rows ride along for free while HBM stays the bottleneck.
Narrow outputs (o_proj, shared expert, hyper-connection down) are split along K across
CTAs; the fp32 partials are reduced and scaled by a second tiny kernel. Wide outputs (qkv,
in_proj, lm_head) need no split and write ``acc * scale`` directly.

Launch config is a pure function of ``(M, N, K)`` (:func:`plan_gemv`), optionally overridden
per shape from a table filled by :func:`tune_gemv` / :func:`set_tuned_plan`. Nothing is
autotuned at launch time, so capturing a CUDA graph around :func:`fp8_dense_linear` is safe:
the first (eager) forward that always precedes a capture compiles the kernels, and a replay
never decides anything.

Large M (prefill): no GEMM is written. The layer's weight is dequantized to a temporary BF16
buffer (scale folded in, one rounding) and handed to the ordinary BF16 matmul, in row slices
so the temporary never exceeds ``_DEQUANT_SLICE_ELEMS``. Layer-major prefill runs 4096-token
chunks, so one dequant (~5 B/element of traffic) is a few percent of the chunk's GEMM.
``torch._scaled_mm`` was rejected: it needs the activation quantized too (a numerics change
that decode does not share), and on sm_89 with torch < 2.12 the row-wise kernel launches off the
current stream (pytorch/pytorch#177651; see ``fp8_pertensor_linear.rowwise_scaled_mm_ok``).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from freetoken.layers.base import BaseOP

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
)

FP8 = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0  # largest finite e4m3 magnitude
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Rows above this go to the dequant + BF16 matmul path. The dot-based kernel stays
# bandwidth-bound while 2*M flop/B of weight stays under the tensor-core roof; past ~32 rows it
# no longer does and the dequantized GEMM wins. FREETOKEN_FP8_GEMV_MAX_M overrides it.
GEMV_MAX_M = int(os.environ.get("FREETOKEN_FP8_GEMV_MAX_M", "32"))
# A dequantized slice is at most this many elements (128 MiB of bf16).
_DEQUANT_SLICE_ELEMS = 64 << 20
# Rows quantized per step at load: bounds the fp32 temporaries to ~256 MiB.
_QUANT_CHUNK_ELEMS = 64 << 20


# ======================================================================================
# Quantization (load time) and the pure-torch reference. Both run on any device.
# ======================================================================================
def quantize_rowwise_fp8(
    w: torch.Tensor, *, chunk_elems: int = _QUANT_CHUNK_ELEMS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row E4M3 quantization of ``w`` ``[N, K]``: ``(q fp8 [N, K], scale fp32 [N])``.

    ``scale = absmax(row) / 448`` so every row spans the full e4m3 range; ``w ~= q * scale``.
    An all-zero row gets scale 1.0 (``q`` is zero either way; a zero scale would divide by
    zero). Row-chunked, so the fp32 temporaries stay bounded for the 1.27 GB lm_head and a
    caller that quantizes layer by layer never holds more than one BF16 weight at a time.
    """
    assert w.dim() == 2, f"expected [N, K], got {tuple(w.shape)}"
    n, k = w.shape
    q = torch.empty((n, k), dtype=FP8, device=w.device)
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    tiny = torch.finfo(torch.float32).tiny
    rows = max(1, chunk_elems // max(k, 1))
    for start in range(0, n, rows):
        block = w[start : start + rows].float()
        amax = block.abs().amax(dim=1)
        s = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax)).clamp_(min=tiny)
        q[start : start + rows] = (block / s[:, None]).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8)
        scale[start : start + rows] = s
    return q, scale


def dequantize_rowwise_fp8(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Inverse of :func:`quantize_rowwise_fp8` (reference / CPU path): ``q * scale[:, None]``."""
    return (q.to(torch.float32) * scale.to(torch.float32)[:, None]).to(dtype)


def fp8_dense_linear_reference(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """The GEMV math, in fp32: ``(x @ q^T) * scale`` with the scale applied to the fp32
    accumulator (exactly where the kernel applies it). The GPU kernel and the dequant-GEMM path
    are compared against this."""
    out = (x.reshape(-1, x.shape[-1]).float() @ q.float().t()) * scale.float()[None, :]
    out = out.reshape(*x.shape[:-1], q.shape[0])
    return out if out_dtype is None else out.to(out_dtype)


# ======================================================================================
# Launch planning: a pure function of the shape, so it is testable without a GPU.
# ======================================================================================
@dataclass(frozen=True)
class GemvPlan:
    """Launch geometry of one GEMV. ``kb_per`` K-blocks per split; ``grid_k`` splits."""

    block_m: int
    block_n: int
    block_k: int
    kb_per: int
    grid_n: int
    grid_k: int
    num_warps: int
    num_stages: int

    @property
    def split_k(self) -> int:
        return self.grid_k


def _block_m(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


# sm_89 allows 99 KiB of shared memory per block; stay a little under it.
_SMEM_BUDGET = 96 * 1024


def fit_stages(block_m: int, block_n: int, block_k: int, num_stages: int) -> int:
    """Deepest pipeline <= ``num_stages`` whose (fp8 weight tile + 16-bit activation tile) per
    stage fits the shared-memory budget. Below 2 the plan is infeasible (see candidate_plans)."""
    per_stage = block_n * block_k + 2 * block_m * block_k
    return min(num_stages, _SMEM_BUDGET // per_stage)


def make_plan(
    m: int, n: int, k: int, *, block_n: int, block_k: int, split_k: int,
    num_warps: int = 4, num_stages: int = 4,
) -> GemvPlan:
    """Plan for an explicit config. ``split_k`` is a request; the realised split is the number
    of non-empty K ranges (``grid_k``), so no program ever starts past the end of K."""
    assert k % 16 == 0 and n >= 1 and block_k % 16 == 0, (n, k, block_k)
    n_kb = triton.cdiv(k, block_k)
    kb_per = triton.cdiv(n_kb, max(1, min(split_k, n_kb)))
    return GemvPlan(
        block_m=_block_m(m), block_n=block_n, block_k=block_k, kb_per=kb_per,
        grid_n=triton.cdiv(n, block_n), grid_k=triton.cdiv(n_kb, kb_per),
        num_warps=num_warps, num_stages=max(2, fit_stages(_block_m(m), block_n, block_k, num_stages)),
    )


# (N, K, block_m) -> (block_n, block_k, split_k, num_warps, num_stages). Filled by tune_gemv /
# set_tuned_plan at warmup (before graph capture); read-only afterwards.
_TUNED: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {}


def set_tuned_plan(
    n: int, k: int, m: int, *, block_n: int, block_k: int, split_k: int,
    num_warps: int = 4, num_stages: int = 4,
) -> None:
    _TUNED[(n, k, _block_m(m))] = (block_n, block_k, split_k, num_warps, num_stages)


def clear_tuned_plans() -> None:
    _TUNED.clear()


def _dividing_block_k(k: int, cap: int = 256) -> int:
    """Largest power-of-two K tile <= ``cap`` that divides ``k`` (>= 16): no K tail, no K mask."""
    bk = cap
    while bk > 16 and k % bk:
        bk //= 2
    return bk


def plan_gemv(m: int, n: int, k: int, *, sm_count: int = 128) -> GemvPlan:
    """Default launch config for an ``[m, k] x [n, k]^T`` GEMV.

    Aim for ~4 resident CTAs per SM: enough independent 8 KB weight tiles in flight to cover HBM
    latency, never fewer than two K-blocks per CTA so the software pipeline has something to
    overlap. Wide outputs get that from N alone and use no split.
    """
    tuned = _TUNED.get((n, k, _block_m(m)))
    if tuned is not None:
        bn, bk, split, warps, stages = tuned
        return make_plan(m, n, k, block_n=bn, block_k=bk, split_k=split,
                         num_warps=warps, num_stages=stages)
    block_k = _dividing_block_k(k)
    block_n = 16 if n <= 4096 else 32
    n_tiles = triton.cdiv(n, block_n)
    n_kb = triton.cdiv(k, block_k)
    split = max(1, min((4 * sm_count) // n_tiles, n_kb // 2))
    return make_plan(m, n, k, block_n=block_n, block_k=block_k, split_k=split)


def emulate_gemv_plan(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, plan: GemvPlan
) -> torch.Tensor:
    """Pure-torch execution of ``plan`` with the kernel's exact tiling, masking and split-K
    reduction order (fp32). Lets the planning arithmetic be checked without a GPU."""
    m, k = x.shape
    n = q.shape[0]
    partial = torch.zeros((plan.grid_k, m, n), dtype=torch.float32)
    for pid_k in range(plan.grid_k):
        k_begin = pid_k * plan.kb_per * plan.block_k
        k_end = min(k, k_begin + plan.kb_per * plan.block_k)
        for pid_n in range(plan.grid_n):
            n0, n1 = pid_n * plan.block_n, min(n, (pid_n + 1) * plan.block_n)
            acc = torch.zeros((m, n1 - n0), dtype=torch.float32)
            for k0 in range(k_begin, k_end, plan.block_k):
                k1 = min(k0 + plan.block_k, k_end)
                acc += x[:, k0:k1].float() @ q[n0:n1, k0:k1].float().t()
            partial[pid_k, :, n0:n1] = acc
    return partial.sum(0) * scale.float()[None, :]


# ======================================================================================
# Decode GEMV kernel
# ======================================================================================
@triton.jit
def _fp8_gemv_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr, part_ptr,
    M, N, K, kb_per,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, DIRECT: tl.constexpr, OUT: tl.constexpr,
):
    """``acc[m, n] = sum_k x[m, k] * w[n, k]`` over this program's K range, fp32 accumulate.

    ``EVEN_K`` (``K % BLOCK_K == 0``, true for every real projection: the planner picks a
    dividing BLOCK_K) drops the K mask so the weight tile is one 16-byte-vector load per thread
    along K; the N mask is constant along K and does not block vectorization.
    ``DIRECT`` (no split): ``out = acc * scale`` is stored here. Otherwise the fp32 partial goes
    to ``part[pid_k, m, n]`` for :func:`_splitk_reduce_kernel`."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_begin = tl.multiple_of(pid_k * kb_per * BLOCK_K, BLOCK_K)
    k_end = tl.minimum(K, k_begin + kb_per * BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        k0 = tl.multiple_of(k0, BLOCK_K)
        offs_k = tl.max_contiguous(tl.multiple_of(k0 + tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
        if EVEN_K:
            x = tl.load(x_ptrs + offs_k[None, :], mask=m_mask[:, None], other=0.0)
            w_mask = n_mask[:, None]
        else:
            k_mask = offs_k < k_end
            x = tl.load(x_ptrs + offs_k[None, :], mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_mask = n_mask[:, None] & k_mask[None, :]
        if e4m3_native_cx():
            w = tl.load(w_ptrs + offs_k[None, :], mask=w_mask, other=0.0).to(x.dtype)
        else:
            w = e4m3_u8_to_f32(
                tl.load(w_ptrs + offs_k[None, :], mask=w_mask, other=0)
            ).to(x.dtype)
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    if DIRECT:
        scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out = (acc * scale[None, :]).to(OUT)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], out,
                 mask=m_mask[:, None] & n_mask[None, :])
    else:
        part = part_ptr + pid_k * M * N
        tl.store(part + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, scale_ptr, out_ptr, M, N, split_k, stride_om,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    pid_m = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(split_k):  # fixed order -> deterministic
        acc += tl.load(part_ptr + (s * M + pid_m) * N + offs, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid_m * stride_om + offs, (acc * scale).to(OUT), mask=mask)


@functools.cache
def _sm_count() -> int:
    if not torch.cuda.is_available():
        return 128
    from freetoken.gpu_select import assigned_visible_gpu

    idx = assigned_visible_gpu()
    return torch.cuda.get_device_properties(
        torch.cuda.current_device() if idx is None else idx
    ).multi_processor_count


def fp8_dense_gemv(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    plan: GemvPlan | None = None,
) -> torch.Tensor:
    """``x [M, K]`` (bf16/fp16, ``M <= GEMV_MAX_M``) against fp8 ``weight [N, K]`` with per-row
    ``weight_scale [N]`` fp32 -> ``[M, N]`` in ``x.dtype``. No host sync, no data-dependent launch."""
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] == k and weight_scale.shape == (n,)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if plan is None:
        plan = plan_gemv(m, n, k, sm_count=_sm_count())
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    direct = plan.grid_k == 1
    part = out if direct else torch.empty((plan.grid_k, m, n), dtype=torch.float32, device=x.device)
    out_tl = _TL_DTYPE[x.dtype]
    _fp8_gemv_kernel[(plan.grid_n, plan.grid_k)](
        x, e4m3_kernel_view(weight), weight_scale, out, part,
        m, n, k, plan.kb_per,
        x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_M=plan.block_m, BLOCK_N=plan.block_n, BLOCK_K=plan.block_k,
        EVEN_K=k % plan.block_k == 0, DIRECT=direct, OUT=out_tl,
        num_warps=plan.num_warps, num_stages=plan.num_stages,
    )
    if not direct:
        _splitk_reduce_kernel[(triton.cdiv(n, 256), m)](
            part, weight_scale, out, m, n, plan.grid_k, out.stride(0),
            BLOCK=256, OUT=out_tl, num_warps=2,
        )
    return out


def candidate_plans(m: int, n: int, k: int, *, sm_count: int = 128) -> list[GemvPlan]:
    """The small fixed config set :func:`tune_gemv` and the microbenchmark sweep: tile widths x
    split targets x two warp counts. Eight to twelve plans, all compiled eagerly at warmup."""
    plans: dict[GemvPlan, None] = {}
    for block_n in (16, 32, 64):
        for block_k in dict.fromkeys(_dividing_block_k(k, cap) for cap in (128, 256, 512)):
            n_tiles = triton.cdiv(n, block_n)
            n_kb = triton.cdiv(k, block_k)
            for per_sm in (2, 4, 8):
                split = max(1, min((per_sm * sm_count) // n_tiles, n_kb // 2))
                for warps in (4, 8):
                    if fit_stages(_block_m(m), block_n, block_k, 4) < 2:
                        continue  # tile too big for sm_89 shared memory even at 2 stages
                    plans[make_plan(m, n, k, block_n=block_n, block_k=block_k,
                                    split_k=split, num_warps=warps)] = None
    return list(plans)


def tune_gemv(
    shapes: Iterable[tuple[int, int]], m: int = 1, *, iters: int = 20, warmup: int = 3,
) -> dict[tuple[int, int, int], GemvPlan]:
    """Pick the fastest candidate plan per ``(N, K)`` by timing it on the GPU and record it with
    :func:`set_tuned_plan`. Call at warmup, before CUDA graph capture, never inside one."""
    assert torch.cuda.is_available()
    dev = torch.cuda.current_device()
    best: dict[tuple[int, int, int], GemvPlan] = {}
    for n, k in dict.fromkeys(shapes):
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w, s = quantize_rowwise_fp8(torch.randn(n, k, device="cuda") * 0.02)
        timed = []
        for plan in candidate_plans(m, n, k, sm_count=_sm_count()):
            try:
                for _ in range(warmup):
                    fp8_dense_gemv(x, w, s, plan)
            except Exception:  # e.g. triton OutOfResources for this tile: not a candidate
                continue
            torch.cuda.synchronize(dev)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fp8_dense_gemv(x, w, s, plan)
            stop.record()
            stop.synchronize()
            timed.append((start.elapsed_time(stop), plan))
        if not timed:
            continue  # keep the default plan for this shape
        _, plan = min(timed, key=lambda t: t[0])
        set_tuned_plan(n, k, m, block_n=plan.block_n, block_k=plan.block_k,
                       split_k=plan.split_k, num_warps=plan.num_warps,
                       num_stages=plan.num_stages)
        best[(n, k, _block_m(m))] = plan
    return best


# ======================================================================================
# Large M: dequantize to BF16 and use the existing matmul
# ======================================================================================
@triton.jit
def _dequant_kernel(
    w_ptr, scale_ptr, out_ptr, N, K, stride_wn, stride_on,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    if e4m3_native_cx():
        w = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        w = e4m3_u8_to_f32(tl.load(ptrs, mask=mask, other=0))
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :],
             (w * scale[:, None]).to(OUT), mask=mask)


def fp8_dequantize(weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``weight * weight_scale[:, None]`` as ``dtype`` in one pass (one rounding)."""
    if not weight.is_cuda:
        return dequantize_rowwise_fp8(weight, weight_scale, dtype)
    n, k = weight.shape
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    _dequant_kernel[(triton.cdiv(n, 32), triton.cdiv(k, 128))](
        e4m3_kernel_view(weight), weight_scale, out, n, k, weight.stride(0), out.stride(0),
        BLOCK_N=32, BLOCK_K=128, OUT=_TL_DTYPE[dtype], num_warps=4,
    )
    return out


def _dequant_matmul(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """``x @ dequant(weight)^T`` through the BF16 matmul, in row slices of the weight."""
    n, k = weight.shape
    rows = max(16, (_DEQUANT_SLICE_ELEMS // k) // 16 * 16)
    if rows >= n:
        return F.linear(x, fp8_dequantize(weight, weight_scale, x.dtype))
    out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        out[:, start:stop] = F.linear(
            x, fp8_dequantize(weight[start:stop], weight_scale[start:stop], x.dtype)
        )
    return out


def fp8_dense_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ (weight_fp8 * weight_scale)^T (+ bias)`` for ``weight`` ``[N, K]`` fp8 and
    ``weight_scale`` ``[N]`` fp32. Small M: the Triton GEMV; larger M (or a CPU tensor): the
    dequantized BF16 matmul."""
    *lead, k = x.shape
    n = weight.shape[0]
    x2 = x.reshape(-1, k)
    if x.is_cuda and x2.shape[0] <= GEMV_MAX_M and k % 16 == 0 and x.dtype in (torch.bfloat16, torch.float16):
        out = fp8_dense_gemv(x2, weight, weight_scale)
    else:
        out = _dequant_matmul(x2, weight, weight_scale)
    out = out.reshape(*lead, n)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ======================================================================================
# BaseOP layers. Buffers: fp8 ``weight`` [out, in] + fp32 ``weight_scale`` [out].
# ======================================================================================
class Fp8DenseLinear(BaseOP):
    """Replicated (TP=1) per-row-FP8 linear: drop-in for ``LinearReplicated`` /
    ``LinearColParallelMerged`` / ``LinearRowParallel`` / ``LinearOProj`` at TP=1. Built by
    :func:`freetoken.models.dense_fp8.apply_dense_fp8` over the BF16 layer it replaces; the
    loader quantizes the checkpoint's BF16 weight into ``weight`` / ``weight_scale``."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale = torch.empty(out_features, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_dense_linear(x, self.weight, self.weight_scale)


class Fp8DenseLMHead(BaseOP):
    """Per-row-FP8 lm_head (untied). Mirrors ``ParallelLMHead.forward`` at TP=1: slice to the last
    token of each sequence at prefill, then the W8A16 projection."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=FP8)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    def forward(self, x: torch.Tensor, *, select_last: bool = True) -> torch.Tensor:
        from freetoken.core import get_global_ctx, select_lm_head_rows

        x = select_lm_head_rows(x, get_global_ctx().batch, select_last=select_last)
        return fp8_dense_linear(x, self.weight, self.weight_scale)


__all__ = [
    "FP8",
    "FP8_E4M3_MAX",
    "GEMV_MAX_M",
    "Fp8DenseLinear",
    "Fp8DenseLMHead",
    "GemvPlan",
    "candidate_plans",
    "clear_tuned_plans",
    "dequantize_rowwise_fp8",
    "emulate_gemv_plan",
    "fp8_dense_gemv",
    "fp8_dense_linear",
    "fp8_dense_linear_reference",
    "fit_stages",
    "fp8_dequantize",
    "make_plan",
    "plan_gemv",
    "quantize_rowwise_fp8",
    "set_tuned_plan",
    "tune_gemv",
]
