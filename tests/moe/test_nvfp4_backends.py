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
import sys
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


def _import_expert_banks_without_kernel_runtime(monkeypatch):
    """Import loader policy helpers without pulling in flashlib's Triton kernels."""
    from freetoken.moe.nvfp4_backends import (
        _NATIVE_NVFP4_BANKS,
        _POST_NVFP4_BANKS,
    )

    module_name = "freetoken.moe.offload_cache"
    fake_offload_cache = types.ModuleType(module_name)
    fake_offload_cache._BANK_BYTES_PER_EXPERT = {}
    fake_offload_cache._BANK_SCHEMAS = {
        "nvfp4": _NATIVE_NVFP4_BANKS,
        "nvfp4_b12x": _POST_NVFP4_BANKS,
        "nvfp4_marlin": _POST_NVFP4_BANKS,
    }
    monkeypatch.setitem(sys.modules, module_name, fake_offload_cache)
    monkeypatch.delitem(sys.modules, "freetoken.moe.expert_banks", raising=False)
    return importlib.import_module("freetoken.moe.expert_banks")


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


def test_moe_activation_dtype_auto_rule_and_explicit_errors():
    from freetoken.moe.nvfp4_backends import resolve_moe_activation_dtype

    assert resolve_moe_activation_dtype(
        "auto", compute_capability=(12, 0), has_input_scales=True
    )[0] == "nvfp4"
    mode, reason = resolve_moe_activation_dtype(
        "auto", compute_capability=(12, 0), has_input_scales=False
    )
    assert mode == "bf16"
    assert "not every" in reason
    mode, reason = resolve_moe_activation_dtype(
        "auto", compute_capability=(8, 9), has_input_scales=True
    )
    assert mode == "bf16"
    assert "sm_120" in reason
    with pytest.raises(RuntimeError, match="requires sm_120"):
        resolve_moe_activation_dtype(
            "nvfp4", compute_capability=(8, 9), has_input_scales=True
        )
    with pytest.raises(RuntimeError, match="lacks input_global_scale"):
        resolve_moe_activation_dtype(
            "nvfp4",
            compute_capability=(12, 0),
            has_input_scales=True,
            b12x_a4_reason="flashinfer b12x_fused_moe lacks input_global_scale",
        )


def test_nvfp4_implicit_backend_waits_for_resolved_activation(monkeypatch):
    from freetoken.moe import nvfp4_backends

    expert_banks = _import_expert_banks_without_kernel_runtime(monkeypatch)
    _resolve_nvfp4_gpu_policy = expert_banks._resolve_nvfp4_gpu_policy

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(nvfp4_backends, "_b12x_a4_unusable_reason", lambda: None)
    calls = []

    def fake_select(_device, _intermediate, requested, **kwargs):
        calls.append((requested, kwargs["prefer_b12x_a4"]))
        return "b12x" if requested == "auto" and kwargs["prefer_b12x_a4"] else "triton"

    monkeypatch.setattr(nvfp4_backends, "select_nvfp4_backend", fake_select)
    device = torch.device("cuda")
    implicit = types.SimpleNamespace(
        nvfp4_backend=None,
        moe_activation_dtype="auto",
        moe_intermediate_size=640,
        hidden_act="silu",
    )
    assert _resolve_nvfp4_gpu_policy(
        implicit, device, has_input_scales=False
    )[:2] == ("triton", "bf16")
    assert calls[-1] == ("triton", False)
    assert _resolve_nvfp4_gpu_policy(
        implicit, device, has_input_scales=True
    )[:2] == ("b12x", "nvfp4")
    assert calls[-1] == ("auto", True)

    explicit_triton = types.SimpleNamespace(**vars(implicit))
    explicit_triton.nvfp4_backend = "triton"
    assert _resolve_nvfp4_gpu_policy(
        explicit_triton, device, has_input_scales=True
    )[:2] == ("triton", "bf16")
    assert calls[-1] == ("triton", False)

    explicit_a4 = types.SimpleNamespace(**vars(implicit))
    explicit_a4.moe_activation_dtype = "nvfp4"
    with pytest.raises(RuntimeError, match="gate/up input scales differ"):
        _resolve_nvfp4_gpu_policy(
            explicit_a4,
            device,
            has_input_scales=False,
            input_scale_unusable_reason="gate/up input scales differ beyond tolerance",
        )


def test_b12x_a4_repack_rejects_unaligned_intermediate():
    from freetoken.moe.nvfp4_backends import b12x_repack_layer

    cfg = types.SimpleNamespace(moe_intermediate_size=192, hidden_size=256)
    with pytest.raises(ValueError, match=r"moe_intermediate_size.*192"):
        b12x_repack_layer({}, cfg, torch.device("cpu"), activation_dtype="nvfp4")


def test_b12x_a4_repack_rejects_unaligned_hidden_size():
    from freetoken.moe.nvfp4_backends import b12x_repack_layer

    cfg = types.SimpleNamespace(moe_intermediate_size=256, hidden_size=192)
    with pytest.raises(ValueError, match=r"hidden_size.*192"):
        b12x_repack_layer({}, cfg, torch.device("cpu"), activation_dtype="nvfp4")


def test_nvfp4_sidecar_load_resolution_skips_bf16_targets(monkeypatch):
    expert_banks = _import_expert_banks_without_kernel_runtime(monkeypatch)
    _nvfp4_load_activation_dtype = expert_banks._nvfp4_load_activation_dtype

    config = types.SimpleNamespace(moe_activation_dtype="auto")
    assert _nvfp4_load_activation_dtype(
        config, torch.device("cpu"), "gpu"
    ) == "bf16"
    assert _nvfp4_load_activation_dtype(
        config, torch.device("cuda"), "cpu"
    ) == "bf16"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    assert _nvfp4_load_activation_dtype(
        config, torch.device("cuda"), "gpu"
    ) == "bf16"


def test_bf16_activation_skips_input_scale_shard_pass(monkeypatch):
    package = types.ModuleType("freetoken.models")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "freetoken.models", package)
    monkeypatch.delitem(sys.modules, "freetoken.models.nvfp4_banks", raising=False)

    source = importlib.util.spec_from_file_location(
        "freetoken.models.nvfp4_banks",
        __file__.replace(
            "tests/moe/test_nvfp4_backends.py",
            "python/freetoken/models/nvfp4_banks.py",
        ),
    )
    assert source is not None and source.loader is not None
    module = importlib.util.module_from_spec(source)
    monkeypatch.setitem(sys.modules, source.name, module)
    source.loader.exec_module(module)

    def fail_drop(_path):
        raise AssertionError("BF16 activation must not visit sidecar shards")

    scales, reason = module._load_input_scales(
        "unused",
        {"sidecar": "must-not-open.safetensors"},
        types.SimpleNamespace(moe_activation_dtype="bf16"),
        object(),
        drop_page_cache=fail_drop,
    )
    assert scales == {}
    assert reason is None


def test_native_ftw_a4_repack_copies_sources_once(monkeypatch):
    from freetoken.checkpoint import ftw
    from freetoken.moe import nvfp4_backends

    expert_banks = _import_expert_banks_without_kernel_runtime(monkeypatch)

    sources = {
        name: [torch.full((2, 2), index, dtype=torch.uint8)]
        for index, name in enumerate(nvfp4_backends._NATIVE_NVFP4_BANKS)
    }
    originals = {name: rows[0].clone() for name, rows in sources.items()}
    loaded = expert_banks.ExpertBanks(
        "nvfp4",
        sources,
        gate_up_input_scale=torch.ones(2),
        down_input_scale=torch.ones(2),
    )
    monkeypatch.setattr(expert_banks, "resolve_bank_source", lambda *_args: "ftw")
    monkeypatch.setattr(ftw, "load_ftw_banks", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        expert_banks,
        "_resolve_nvfp4_gpu_policy",
        lambda *_args, **_kwargs: ("b12x", "nvfp4", "test A4 policy"),
    )
    calls = []

    def fake_repack(copied, *_args, **kwargs):
        calls.append(copied)
        assert kwargs["activation_dtype"] == "nvfp4"
        for name, rows in copied.items():
            assert rows[0].data_ptr() != sources[name][0].data_ptr()
            rows[0].add_(1)
        copied["gate_up_alpha"] = torch.ones(2)
        copied["down_alpha"] = torch.ones(2)
        return copied

    monkeypatch.setattr(
        nvfp4_backends, "b12x_repack_sources_inplace", fake_repack
    )
    config = types.SimpleNamespace(num_moe_layers=1)
    result = expert_banks._load_expert_banks_impl(
        "unused",
        config,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        decode_target="gpu",
    )

    assert len(calls) == 1
    assert result.quant_format == "nvfp4_b12x"
    assert result.activation_dtype == "nvfp4"
    for name, original in originals.items():
        torch.testing.assert_close(sources[name][0], original)


def test_native_ftw_rejects_selected_w4a16_backend(monkeypatch):
    from freetoken.checkpoint import ftw

    expert_banks = _import_expert_banks_without_kernel_runtime(monkeypatch)

    loaded = expert_banks.ExpertBanks(
        "nvfp4", {"gate_up_packed": [torch.zeros(1, dtype=torch.uint8)]}
    )
    monkeypatch.setattr(expert_banks, "resolve_bank_source", lambda *_args: "ftw")
    monkeypatch.setattr(ftw, "load_ftw_banks", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        expert_banks,
        "_resolve_nvfp4_gpu_policy",
        lambda *_args, **_kwargs: ("marlin", "bf16", "explicit marlin request"),
    )

    with pytest.raises(RuntimeError, match=r"selected 'marlin'.*reconvert"):
        expert_banks._load_expert_banks_impl(
            "unused",
            types.SimpleNamespace(num_moe_layers=1, nvfp4_backend="marlin"),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            decode_target="gpu",
        )


def test_b12x_a4_folds_fc2_input_scale_into_weight_views(monkeypatch):
    from freetoken.moe import nvfp4_backends

    observed = {}

    def fake_weight_views(*args):
        observed["folded_down_alpha"] = args[5]
        return object()

    def fake_launch(**kwargs):
        observed["launch_w2_alpha"] = kwargs["w2_alpha"]
        return kwargs["scatter_output"]

    dispatch_name = (
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
    )
    monkeypatch.setitem(
        sys.modules,
        dispatch_name,
        types.SimpleNamespace(launch_sm120_moe=fake_launch),
    )
    monkeypatch.setattr(nvfp4_backends, "_b12x_a4_weight_views", fake_weight_views)

    experts = 3
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    gate_up_q = torch.zeros((experts, 2, 2), dtype=torch.uint8)
    gate_up_s = torch.zeros_like(gate_up_q)
    down_q = torch.zeros_like(gate_up_q)
    down_s = torch.zeros_like(gate_up_q)
    gate_up_alpha = torch.stack((torch.full((experts,), 2.0), torch.full((experts,), 0.25)))
    down_alpha = torch.stack((torch.full((experts,), 3.0), torch.full((experts,), 0.125)))

    nvfp4_backends.b12x_fused_experts(
        hidden,
        gate_up_q,
        gate_up_s,
        gate_up_alpha,
        down_q,
        down_s,
        down_alpha,
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        "silu",
        False,
    )

    torch.testing.assert_close(
        observed["folded_down_alpha"], down_alpha[0] * down_alpha[1]
    )
    torch.testing.assert_close(observed["launch_w2_alpha"], down_alpha[0])


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
def test_b12x_nvfp4_activation_parity_with_bf16_activation_path():
    """SM120 only: W4A4 stays within the fixed ModelOpt-style parity gate.

    The 12 percent relative and 8 percent peak-magnitude absolute tolerances account
    for two additional E2M1 quantization boundaries, at the FC1 input and before FC2.
    They are intentionally fixed before the first G4 run.
    """
    from freetoken.moe.nvfp4_backends import (
        _b12x_a4_unusable_reason,
        _b12x_unusable_reason,
        b12x_fused_experts,
        b12x_repack_sources_inplace,
    )

    device = torch.device("cuda")
    cc = torch.cuda.get_device_capability(device)
    reason = _b12x_unusable_reason(cc) or _b12x_a4_unusable_reason()
    if cc != (12, 0) or reason is not None:
        pytest.skip(f"SM120 b12x NVFP4 path unavailable: {reason or cc}")

    native = _make_native_sources(device, seed=18)
    w4a16_sources = {name: [row.clone() for row in rows] for name, rows in native.items()}
    w4a4_sources = {name: [row.clone() for row in rows] for name, rows in native.items()}
    cfg = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I)
    w4a16 = b12x_repack_sources_inplace(
        w4a16_sources, cfg, device, chunk=E, activation_dtype="bf16"
    )
    w4a4 = b12x_repack_sources_inplace(
        w4a4_sources, cfg, device, chunk=E, activation_dtype="nvfp4"
    )

    torch.manual_seed(19)
    hidden = torch.randn(8, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (8, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(8, TOPK, dtype=torch.float32, device=device)
    gate_input = torch.full((E,), 0.25, dtype=torch.float32, device=device)
    down_input = torch.full((E,), 0.125, dtype=torch.float32, device=device)

    ref = b12x_fused_experts(
        hidden,
        w4a16["gate_up_packed"][0], w4a16["gate_up_scale"][0],
        w4a16["gate_up_alpha"][:E],
        w4a16["down_packed"][0], w4a16["down_scale"][0],
        w4a16["down_alpha"][:E],
        topk_weights, topk_ids, "silu", False,
    )
    out = b12x_fused_experts(
        hidden,
        w4a4["gate_up_packed"][0], w4a4["gate_up_scale"][0],
        torch.stack((w4a4["gate_up_alpha"][:E], gate_input)),
        w4a4["down_packed"][0], w4a4["down_scale"][0],
        torch.stack((w4a4["down_alpha"][:E], down_input)),
        topk_weights, topk_ids, "silu", False,
    )
    atol = 0.08 * max(float(ref.float().abs().max()), 1e-3)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.12, atol=atol)


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
