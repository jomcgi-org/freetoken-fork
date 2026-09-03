"""Numerical tests for the NVFP4 MoE backends (Triton inline-dequant / Marlin / b12x).

Each verifies the full chain ``native banks -> (in-place repack) -> offload cache slot
gather -> fused forward`` against a pure-torch dequant reference, for both regimes (decode
routes slot ids into the full cache; full-layer prefill routes raw expert ids into the
materialized ``[:E]`` view or the overlap double-buffer views).

Coverage by hardware:
  - Triton (any CUDA GPU): prefill + the production fast decode GEMV, plus a fast-vs-
    baseline-kernel equality guard. This is the path used on sm_120 + CUDA 12.x.
  - Marlin (sm_80..sm_99, e.g. H100): prefill + decode + overlap.
  - b12x (sm_120 + CUDA>=13): pure-torch pack everywhere; the fused decode forward is
    gated and skipped where the kernel cannot run.
The ``--nvfp4-backend`` selection + CUDA-13 gate is checked without a GPU.
"""

from __future__ import annotations

import importlib.util
import types

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# vllm (marlin W4A16 path) is intentionally not co-installable with the core transformers
# pin; it lives in a dedicated venv. Skip rather than fail.
marlin = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="needs vllm (marlin path)",
)

L, E, S = 2, 8, 8  # layers, experts/layer, cache slots
H, I = 256, 128  # hidden, moe intermediate
TOPK = 2

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _dequant_ref(packed: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    """[N, K//2] u8 + [N, K//16] e4m3 + [N] global -> [N, K] fp32 (low nibble first)."""
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    w = _E2M1.to(packed.device)[codes]
    s = scale.float().repeat_interleave(16, dim=1)
    return w * s * row_global.float().unsqueeze(1)


def _make_native_sources(device: torch.device, seed: int = 0) -> dict[str, list[torch.Tensor]]:
    """Random ModelOpt-style banks, CPU pinned, with one expert whose w1/w3 globals differ.

    One flat ``[L*E, ...]`` RNG draw (so seeding is unaffected) split into L
    per-layer views.
    """
    g = torch.Generator().manual_seed(seed)
    total = L * E

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    gate_up_global = torch.full((total, 2 * I), 1.0, dtype=torch.float16)
    gate_up_global[:, I:] = 0.5  # w3 global != w1 global: exercises the alpha fold
    down_global = torch.full((total, H), 0.75, dtype=torch.float16)
    flat = {
        "gate_up_packed": rand_u8(total, 2 * I, H // 2),
        "gate_up_scale": rand_scale(total, 2 * I, H // 16),
        "gate_up_global": gate_up_global,
        "down_packed": rand_u8(total, H, I // 2),
        "down_scale": rand_scale(total, H, I // 16),
        "down_global": down_global,
    }
    return {name: list(t.pin_memory().split(E)) for name, t in flat.items()}


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    """bf16 grouped GEMMs round the (large) gate_up intermediates to bf16, so the
    achievable accuracy is relative to the output magnitude, not absolute."""
    tol = 0.03 * float(ref.abs().max())
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=tol)


def _swigluoai_ref(h: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """MiniMax-M3 / gpt-oss clamped swiglu over UNINTERLEAVED [gate; up] halves."""
    gate = h[:I].clamp(max=limit)
    up = h[I:].clamp(-limit, limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="silu") -> torch.Tensor:
    """Dequant + dense per-token reference for the gated MoE (silu or swigluoai)."""
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    for t in range(hidden.size(0)):
        for j in range(topk_ids.size(1)):
            e = int(topk_ids[t, j])
            gu = _dequant_ref(
                sources["gate_up_packed"][layer_id][e].to(hidden.device),
                sources["gate_up_scale"][layer_id][e].to(hidden.device),
                sources["gate_up_global"][layer_id][e].to(hidden.device),
            )
            dn = _dequant_ref(
                sources["down_packed"][layer_id][e].to(hidden.device),
                sources["down_scale"][layer_id][e].to(hidden.device),
                sources["down_global"][layer_id][e].to(hidden.device),
            )
            h = gu @ x[t]
            if activation == "swigluoai":
                act = _swigluoai_ref(h)
            else:
                act = torch.nn.functional.silu(h[:I]) * h[I:]
            out[t] += float(topk_weights[t, j]) * (dn @ act)
    return out


def _marlin_cache(device, *, cache_size=S, prefill_overlap=False):
    from freetoken.moe.nvfp4_backends import marlin_repack_sources_inplace
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device)
    ref_sources = {k: [t.clone() for t in v] for k, v in sources.items()}  # repack is in place
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    packed = marlin_repack_sources_inplace(sources, cfg, device, chunk=5)

    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format="nvfp4_marlin",
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()
    return cache, ref_sources


@cuda
@marlin
def test_marlin_prefill_matches_dequant_reference():
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(1)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, topk_ids)

    # Synchronous full-layer prefill: slot == expert id, raw routing ids pass through.
    cache.materialize_layer(0)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_layer(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views(E)
    out = marlin_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        topk_weights, topk_ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
@marlin
def test_marlin_decode_matches_dequant_reference_after_prefill_stomp():
    """Decode through the slot cache, including the request-B-after-request-A pattern
    that B1 guarded against: a layer-1 full-layer prefill between two layer-0 decodes."""
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        g1, g2 = cache.alphas_for_slots(layer_id)
        gu_p, gu_s, dn_p, dn_s = cache.bank_views()
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, ids, "silu", False,
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
@marlin
def test_marlin_overlap_prefill_matches_dequant_reference():
    """prefill_overlap=True over NVFP4 banks: every layer streams through the generic
    double buffer (full-layer views, routing ids unmapped), and a decode afterwards is
    still correct -- the prefetch invalidated the bookkeeping of the stomped slots.

    The decode-after check is armed by claiming layer-0 slots *before* the prefill
    (cache_size == 2E, so every slot is buffer-backed): if the prefetch failed to
    invalidate them, the post-prefill decode would "hit" stale mappings and read other
    experts' bytes; we assert it misses both experts instead."""
    from freetoken.moe.nvfp4_backends import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device, cache_size=2 * E, prefill_overlap=True)
    torch.manual_seed(4)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    warm_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, warm_ids)
    cache.copy_missing()

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, dn_p, dn_s = cache.wait_prefill_layer(layer_id)
        g1, g2 = cache.alphas_for_layer(layer_id)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, topk_ids)
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, topk_ids, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)

    # Decode the pre-claimed experts after the buffers stomped the whole cache: their
    # old slot mappings must be gone (forced miss + reload), not "hit" stale entries
    # now holding other layers' prefill bytes.
    dec_hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    dec_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, dec_hidden, dec_weights, ids)
    cache.ensure_experts(0, ids)
    assert int(cache.num_indices.item()) == 2, "stale slot mappings survived the prefetch"
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views()
    out = marlin_fused_experts(
        dec_hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        dec_weights, ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
def test_triton_overlap_prefill_matches_dequant_reference():
    """The 6-bank native layout through the same generic double buffer, consumed by
    the Triton inline-dequant grouped GEMM with unmapped routing ids (n = E)."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=5)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=2 * E,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=True,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    torch.manual_seed(6)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.wait_prefill_layer(layer_id)
        ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids)
        out = fused_experts_nvfp4(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
            topk_weights, topk_ids, E, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)


@cuda
def test_triton_swigluoai_matches_dequant_reference():
    """MiniMax-M3's swigluoai routed experts through the Triton prefill grouped GEMM
    and the marlin-style decode GEMV: same banks, the clamped (up+1) swiglu instead
    of silu, alpha/limit threaded through the fused entry points."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=11)
    torch.manual_seed(12)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)
    layer_id = 0
    banks = [
        sources[name][layer_id].to(device)
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    ]
    ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="swigluoai")
    out = fused_experts_nvfp4(
        hidden, *banks, topk_weights, topk_ids, E, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)

    dec_hidden = hidden[:1]
    dec_ids = topk_ids[:1]
    dec_weights = topk_weights[:1]
    ref = _ref_moe(sources, layer_id, dec_hidden, dec_weights, dec_ids, activation="swigluoai")
    out = fused_experts_decode_nvfp4_marlin(
        dec_hidden, *banks, dec_weights, dec_ids, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)


def _triton_cache(device, *, cache_size=S, prefill_overlap=False):
    """Native 6-bank NVFP4 cache (no repack), consumed directly by the Triton kernels.
    The banks are not transformed, so ``sources`` doubles as the dequant reference."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device, seed=7)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    return cache, sources


@cuda
def test_triton_decode_marlin_matches_dequant_reference_after_prefill_stomp():
    """The production marlin-style int32 decode GEMV through the slot cache, including the
    request-B-after-request-A pattern (a layer-1 full-layer prefill between two layer-0
    decodes) that must force a miss + reload rather than serve stale slot bytes."""
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

    device = torch.device("cuda")
    cache, ref_sources = _triton_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.bank_views()
        out = fused_experts_decode_nvfp4_marlin(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_weights, ids, "silu", False
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
def test_triton_decode_marlin_matches_baseline_kernel():
    """The production marlin-style decode GEMV must match the original LUT-gather decode
    within tolerance (it only reorders the dequant math: int32 wide load + deferred reduce)."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_decode_nvfp4_serial,
    )

    device = torch.device("cuda")
    cache, _ = _triton_cache(device)
    torch.manual_seed(11)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[1, 6]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    banks = cache.bank_views()
    marlin = fused_experts_decode_nvfp4_marlin(hidden, *banks, topk_weights, ids, "silu", False)
    base = fused_experts_decode_nvfp4_serial(hidden, *banks, topk_weights, ids, "silu", False)
    torch.testing.assert_close(marlin.float(), base.float(), rtol=2e-3, atol=2e-3)


REAL_EXPERTS = 512
REAL_CACHE_SIZE = 2564
REAL_HOT_ROWS = 55


def test_nvfp4_prefill_real_slot_table_bounds():
    from freetoken.moe.fused_nvfp4 import _assert_prefill_table_bounds

    ids = torch.tensor([[1024, REAL_CACHE_SIZE - 1]], dtype=torch.int32)
    tables = tuple(
        (name, torch.empty((REAL_CACHE_SIZE, 0)))
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    )
    _assert_prefill_table_bounds(ids, REAL_CACHE_SIZE, tables)

    layer_sized_global = (
        *tables[:2],
        ("gate_up_global", torch.empty((REAL_EXPERTS, 0))),
        *tables[3:],
    )
    with pytest.raises(
        AssertionError,
        match=r"gate_up_global has 512 rows, expected 2564",
    ):
        _assert_prefill_table_bounds(ids, REAL_CACHE_SIZE, layer_sized_global)


def test_align_active_slots_preserves_production_slot_routing(monkeypatch):
    import freetoken.moe.fused_nvfp4 as fused_nvfp4

    resident_slots = torch.linspace(
        1025, REAL_CACHE_SIZE - 1, REAL_HOT_ROWS, dtype=torch.float32
    ).round().to(torch.int32).unique()
    assert resident_slots.numel() == REAL_HOT_ROWS
    assert int(resident_slots.min()) > 2 * REAL_EXPERTS
    route_count = 21 * 8
    ids = resident_slots[torch.arange(route_count) % REAL_HOT_ROWS].reshape(21, 8)
    block_size = 32
    align_calls = []

    def reference_align(dense_ids, requested_block_size, num_experts):
        align_calls.append((dense_ids.clone(), requested_block_size, num_experts))
        flat_ids = dense_ids.reshape(-1).tolist()
        padding_id = len(flat_ids)
        sorted_ids = []
        expert_ids = []
        for expert_id in range(num_experts):
            routed = [i for i, value in enumerate(flat_ids) if value == expert_id]
            padded = ((len(routed) + block_size - 1) // block_size) * block_size
            sorted_ids.extend(routed)
            sorted_ids.extend([padding_id] * (padded - len(routed)))
            expert_ids.extend([expert_id] * (padded // block_size))
        ntpp = len(sorted_ids)
        # Match the production aligner's spare capacity with values outside the
        # valid prefix. _align_active_slots must map it without indexing errors.
        expert_ids.extend([-1, num_experts + 1])
        return (
            torch.tensor(sorted_ids, dtype=torch.int32),
            torch.tensor(expert_ids, dtype=torch.int32),
            torch.tensor([ntpp], dtype=torch.int32),
        )

    monkeypatch.setattr(fused_nvfp4, "moe_align_block_size", reference_align)
    monkeypatch.setattr(fused_nvfp4, "is_sgl_kernel_installed", lambda: True)

    sorted_ids, expert_slots, ntpp = fused_nvfp4._align_active_slots(
        ids, block_size
    )

    dense_ids, requested_block_size, active_count = align_calls.pop()
    assert requested_block_size == block_size
    assert active_count == REAL_HOT_ROWS
    assert set(dense_ids.reshape(-1).tolist()) == set(range(REAL_HOT_ROWS))

    valid_count = int(ntpp[0])
    valid_routes = sorted_ids[:valid_count]
    valid_routes = valid_routes[valid_routes < route_count]
    assert valid_routes.numel() == route_count
    assert valid_routes.unique().numel() == route_count
    assert sorted(valid_routes.tolist()) == list(range(route_count))

    flat_original_ids = ids.reshape(-1)
    mapped_slots = []
    for block_index in range(valid_count // block_size):
        block_routes = sorted_ids[
            block_index * block_size : (block_index + 1) * block_size
        ]
        block_routes = block_routes[block_routes < route_count].long()
        mapped_slot = int(expert_slots[block_index])
        assert block_routes.numel() > 0
        assert (flat_original_ids[block_routes] == mapped_slot).all()
        mapped_slots.append(mapped_slot)

    assert len(mapped_slots) == len(set(mapped_slots))
    assert set(mapped_slots) == set(resident_slots.tolist())
    assert set(mapped_slots).issubset(set(resident_slots.tolist()))


def test_nvfp4_full_layer_prefill_keeps_direct_alignment_path(monkeypatch):
    import freetoken.moe.fused_nvfp4 as fused_nvfp4

    ids = torch.tensor([[0, REAL_EXPERTS - 1]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    direct_calls = []

    def direct_align(topk_ids, block_size, num_experts):
        direct_calls.append((topk_ids, block_size, num_experts))
        return (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )

    monkeypatch.setattr(fused_nvfp4, "moe_align_block_size", direct_align)
    monkeypatch.setattr(
        fused_nvfp4,
        "_align_active_slots",
        lambda *_args: pytest.fail("full-layer prefill compacted active slots"),
    )
    monkeypatch.setattr(
        fused_nvfp4,
        "_assert_prefill_table_bounds",
        lambda *_args: pytest.fail("full-layer prefill synchronized table bounds"),
    )
    monkeypatch.setattr(fused_nvfp4, "_prefill_gemm", lambda *_args: None)
    monkeypatch.setattr(fused_nvfp4, "_run_act", lambda *_args: None)
    monkeypatch.setattr(
        fused_nvfp4, "moe_sum_reduce_triton", lambda _source, out: out.zero_()
    )

    gate_up_packed = torch.empty((REAL_EXPERTS, 6, 2), dtype=torch.uint8)
    gate_up_scale = torch.empty((REAL_EXPERTS, 6, 1), dtype=torch.float8_e4m3fn)
    gate_up_global = torch.empty((REAL_EXPERTS, 6), dtype=torch.float16)
    down_packed = torch.empty((REAL_EXPERTS, 4, 2), dtype=torch.uint8)
    down_scale = torch.empty((REAL_EXPERTS, 4, 1), dtype=torch.float8_e4m3fn)
    down_global = torch.empty((REAL_EXPERTS, 4), dtype=torch.float16)
    fused_nvfp4.fused_experts_nvfp4(
        hidden,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        down_packed,
        down_scale,
        down_global,
        weights,
        ids,
        REAL_EXPERTS,
    )

    assert len(direct_calls) == 1
    assert direct_calls[0][0] is ids
    assert direct_calls[0][1:] == (16, REAL_EXPERTS)


def _real_slot_nvfp4_banks(device):
    """Small-feature NVFP4 tables with the production row geometry."""
    generator = torch.Generator().manual_seed(71)

    def rand_scale(*shape):
        return (
            torch.rand(*shape, generator=generator) * 1.5 + 0.25
        ).to(torch.float8_e4m3fn)

    slots = torch.linspace(
        1024, REAL_CACHE_SIZE - 1, REAL_HOT_ROWS, dtype=torch.float32
    ).round().to(torch.long).unique()
    assert slots.numel() == REAL_HOT_ROWS
    compact = {
        "gate_up_packed": torch.randint(
            0, 256, (REAL_HOT_ROWS, 2 * I, H // 2),
            dtype=torch.uint8, generator=generator,
        ),
        "gate_up_scale": rand_scale(REAL_HOT_ROWS, 2 * I, H // 16),
        "down_packed": torch.randint(
            0, 256, (REAL_HOT_ROWS, H, I // 2),
            dtype=torch.uint8, generator=generator,
        ),
        "down_scale": rand_scale(REAL_HOT_ROWS, H, I // 16),
    }
    gate_up_global = torch.full(
        (REAL_HOT_ROWS, 2 * I), 1.0, dtype=torch.float16
    )
    gate_up_global[:, I:] = 0.5
    compact["gate_up_global"] = gate_up_global
    compact["down_global"] = torch.full(
        (REAL_HOT_ROWS, H), 0.75, dtype=torch.float16
    )
    full = []
    for name in (
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    ):
        source = compact[name]
        table = torch.empty(
            (REAL_CACHE_SIZE, *source.shape[1:]), dtype=source.dtype, device=device
        )
        # index_copy_ is not implemented for float8 on CUDA; the byte view
        # keeps the row axis and copies every dtype the same way.
        table.view(torch.uint8).index_copy_(
            0, slots.to(device), source.to(device).view(torch.uint8)
        )
        full.append(table)
    return slots, compact, tuple(full)


def _hot_partial_routing(tokens, slots, device):
    total = tokens * 8
    hot_routes = int(total * 0.55)
    flat_ids = torch.full((total,), slots[0], dtype=slots.dtype)
    flat_ids[:hot_routes] = slots[torch.arange(hot_routes) % slots.numel()]
    flat_weights = torch.rand(total, generator=torch.Generator().manual_seed(72))
    flat_weights[hot_routes:] = 0
    return (
        flat_ids.reshape(tokens, 8).to(device=device, dtype=torch.int32).contiguous(),
        flat_weights.reshape(tokens, 8).to(device=device, dtype=torch.float32).contiguous(),
    )


@cuda
def test_grouped_hot_partial_real_slot_geometry_matches_decode_and_cpu():
    """Run with CUDA_LAUNCH_BLOCKING=1 to attribute any invalid access to its launch."""
    from freetoken.moe.fused_nvfp4 import (
        _align_active_slots,
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    device = torch.device("cuda")
    slots, compact, banks = _real_slot_nvfp4_banks(device)
    tokens = 13
    ids, weights = _hot_partial_routing(tokens, slots, device)
    hidden = (
        torch.randn(tokens, H, generator=torch.Generator().manual_seed(73)) / 16
    ).to(device=device, dtype=torch.bfloat16)
    assert int(ids.max().item()) == REAL_CACHE_SIZE - 1
    assert int(ids.max().item()) > REAL_EXPERTS

    grouped = fused_experts_nvfp4(
        hidden, *banks, weights, ids, REAL_CACHE_SIZE, "silu", False
    )
    decoded = fused_experts_decode_nvfp4_marlin(
        hidden, *banks, weights, ids, "silu", False
    )

    active_slots = torch.unique(ids.cpu(), sorted=True)
    slot_to_compact = {int(slot): row for row, slot in enumerate(slots.tolist())}
    compact_ids = torch.tensor(
        [slot_to_compact[int(slot)] for slot in ids.cpu().reshape(-1)],
        dtype=torch.int32,
    ).reshape_as(ids.cpu())
    sources = {name: [table] for name, table in compact.items()}
    cpu_ref = _ref_moe(
        sources, 0, hidden.cpu(), weights.cpu(), compact_ids
    )
    torch.cuda.synchronize()

    _assert_close(grouped, cpu_ref.to(device))
    _assert_close(decoded, cpu_ref.to(device))
    _assert_close(grouped, decoded)
    _sorted, expert_slots, ntpp = _align_active_slots(ids, 32)
    valid_blocks = int(ntpp.item()) // 32
    assert set(expert_slots[:valid_blocks].cpu().tolist()) == set(active_slots.tolist())


@cuda
def test_grouped_hot_partial_real_geometry_timing(capsys):
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    device = torch.device("cuda")
    slots, _compact, banks = _real_slot_nvfp4_banks(device)
    tokens = 1800
    ids, weights = _hot_partial_routing(tokens, slots, device)
    hidden = (
        torch.randn(tokens, H, generator=torch.Generator().manual_seed(74)) / 16
    ).to(device=device, dtype=torch.bfloat16)

    def grouped():
        return fused_experts_nvfp4(
            hidden, *banks, weights, ids, REAL_CACHE_SIZE, "silu", False
        )

    def decoded():
        return fused_experts_decode_nvfp4_marlin(
            hidden, *banks, weights, ids, "silu", False
        )

    def elapsed_ms(run):
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 5

    grouped_ms = elapsed_ms(grouped)
    decode_ms = elapsed_ms(decoded)
    ratio = decode_ms / grouped_ms
    with capsys.disabled():
        print(
            "real-slot hot partial per layer: "
            f"grouped={grouped_ms:.3f} ms decode={decode_ms:.3f} ms "
            f"decode/grouped={ratio:.3f}x"
        )
    assert grouped_ms > 0 and decode_ms > 0


def test_nvfp4_backend_selection():
    """--nvfp4-backend selection + the flashinfer/marlin device gates -- runs without a GPU
    via the CPU branch (forced backends need a usable device, so they error loudly there)."""
    from freetoken.moe.nvfp4_backends import select_nvfp4_backend

    cpu = torch.device("cpu")
    assert select_nvfp4_backend(cpu, None, "triton") == "triton"
    assert select_nvfp4_backend(cpu, None, "auto") == "triton"  # auto on CPU
    with pytest.raises(RuntimeError):
        select_nvfp4_backend(cpu, None, "flashinfer")  # b12x needs a CUDA device
    with pytest.raises(RuntimeError):
        select_nvfp4_backend(cpu, None, "marlin")  # marlin needs a CUDA device
    with pytest.raises(ValueError):
        select_nvfp4_backend(cpu, None, "bogus")


@cuda
def test_b12x_decode_matches_dequant_reference():
    """sm_120 + CUDA>=13 only: the flashinfer b12x W4A16 fused MoE over the slot cache
    vs the dequant reference (skipped on hardware/toolkits where b12x cannot run)."""
    from freetoken.moe.nvfp4_backends import (
        _b12x_unusable_reason,
        b12x_fused_experts,
        b12x_repack_sources_inplace,
    )
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
    if reason is not None:
        pytest.skip(f"b12x not runnable here: {reason}")

    sources = _make_native_sources(device, seed=8)
    ref_sources = {k: [t.clone() for t in v] for k, v in sources.items()}  # repack is in place
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    packed = b12x_repack_sources_inplace(sources, cfg, device, chunk=6)

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=device, quant_format="nvfp4_b12x"
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()

    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, ids)

    cache.ensure_experts(0, ids)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.bank_views()
    out = b12x_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2, topk_weights, ids, "silu", False
    )
    _assert_close(out, ref)


@cuda
def test_dummy_nvfp4_sources_match_loader_contract():
    """--use-dummy-weight banks must match the real loader's shapes/dtypes/pinning so the
    engine repack/offload path is exercised unchanged. The marlin repack + offload gather
    tail (which needs vllm) lives in test_dummy_nvfp4_sources_marlin_repack."""
    from freetoken.models.weight import dummy_nvfp4_expert_sources

    cfg = types.SimpleNamespace(
        num_layers=L, num_experts=E, hidden_size=H, moe_intermediate_size=I
    )
    sources = dummy_nvfp4_expert_sources(cfg)
    expected = {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), torch.float8_e4m3fn),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), torch.float8_e4m3fn),
        "down_global": ((E, H), torch.float16),
    }
    assert sources.keys() == expected.keys()
    for name, (shape, dtype) in expected.items():
        layers = sources[name]
        assert len(layers) == L, (name, len(layers))
        for t in layers:
            assert t.shape == shape and t.dtype == dtype and t.is_pinned(), name


@cuda
@marlin
def test_dummy_nvfp4_sources_marlin_repack():
    """The --use-dummy-weight banks drop into the same marlin repack + offload path as the
    real loader's (in-place repack). The gather kernel reads the banks zero-copy from the
    GPU, which requires the allocator's memory to be device-mapped, not merely page-locked."""
    from freetoken.models.weight import dummy_nvfp4_expert_sources
    from freetoken.moe.nvfp4_backends import marlin_repack_sources_inplace
    from freetoken.moe.offload_cache import OffloadMoeCache

    cfg = types.SimpleNamespace(
        num_layers=L, num_experts=E, hidden_size=H, moe_intermediate_size=I
    )
    sources = dummy_nvfp4_expert_sources(cfg)

    device = torch.device("cuda")
    packed = marlin_repack_sources_inplace(sources, cfg, device, chunk=5)
    assert torch.isfinite(packed["gate_up_alpha"].float()).all()
    assert torch.isfinite(packed["down_alpha"].float()).all()

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=device, quant_format="nvfp4_marlin"
    )
    cache.set_bank_sources({name: packed[name] for name in cache.bank_schema})
    cache.set_alphas(packed["gate_up_alpha"], packed["down_alpha"])
    cache.reset()
    cache.materialize_layer(0)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert torch.equal(cache.bank_caches["gate_up_packed"][:E].cpu(), packed["gate_up_packed"][0])


@cuda
@pytest.mark.slow
def test_b12x_pack_is_byte_compatible_with_native_banks():
    """The b12x kernel needs sm_120, but its pack is pure torch: verify the prepared
    blocks drop into the native banks byte-for-byte (the in-place repack contract)."""
    from freetoken.moe.nvfp4_backends import b12x_repack_sources_inplace

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=3)
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    try:
        packed = b12x_repack_sources_inplace(sources, cfg, device, chunk=6)
    except Exception as exc:  # pragma: no cover - depends on flashinfer internals
        pytest.skip(f"flashinfer w4a16 prepare unavailable off-target: {exc}")
    total = L * E
    # packed banks stay per-layer lists; alphas are the one flat [L*E] exception (see
    # cache_budget.expert_bytes_per_slot).
    assert len(packed["gate_up_packed"]) == L
    assert sum(t.shape[0] for t in packed["gate_up_packed"]) == total
    assert packed["gate_up_alpha"].shape == (total,)
    assert packed["down_packed"][0].dtype == torch.int32
