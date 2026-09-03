"""Per-quant-format expert bank providers for the offload MoE cache.

A provider owns everything between "checkpoint on disk" and "banks ready for
``OffloadMoeCache.set_bank_sources``" for one ``expert_quant`` value: loading the
pinned host banks, picking a kernel backend, repacking into its layout, and
extracting the GPU-resident alphas if the format folds its global scales. The
engine stays quant-agnostic: it calls :func:`load_expert_banks` and wires the
returned bundle into the cache.

A format is fully described by three per-format tables: ``_BANK_SCHEMAS``
(offload_cache, bank layout), ``_PROVIDERS`` (here, loading/repack), and
``OffloadMoELayer._expert_gemm`` (kernel dispatch). Adding a format touches those
three places and nothing else.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import threading
from dataclasses import dataclass, field, replace

import torch

from freetoken.utils import init_logger

from .offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS

logger = init_logger(__name__)

# the parallel expert-bank reader needs POSIX O_DIRECT + preadv; without them the serial (safetensors/mmap) build is the only option
_PARALLEL_READER_SUPPORTED = hasattr(os, "O_DIRECT") and hasattr(os, "preadv")


def resolve_bank_source(model_path: str | None, requested: str = "auto") -> str:
    """Resolve to ``ftw``, ``index``, or the legacy materializing ``source`` path."""
    if requested not in ("auto", "ftw", "index"):
        raise ValueError(f"unknown bank source {requested!r}")
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    ftw = bool(model_path and is_ftw_checkpoint(model_path))
    if ftw:
        if requested == "index":
            raise ValueError("--bank-source index requires a safetensors checkpoint")
        return "ftw"
    if requested == "ftw":
        raise ValueError("--bank-source ftw requires an FTW checkpoint")
    if not model_path:
        if requested == "index":
            raise ValueError("--bank-source index requires a checkpoint path")
        return "source"

    from freetoken.checkpoint.safetensors_bank_index import (
        UnsupportedSafetensorsBankIndex,
        ensure_safetensors_bank_index,
    )

    try:
        ensure_safetensors_bank_index(model_path)
    except (FileNotFoundError, UnsupportedSafetensorsBankIndex):
        if requested == "index":
            raise
        return "source"
    return "index"


@dataclass(frozen=True)
class ExpertBanks:
    """Loaded expert banks, normalized for ``OffloadMoeCache`` wiring."""

    quant_format: str  # _BANK_SCHEMAS key
    # Pinned host banks, keyed by the format's schema: one [num_experts, ...]
    # tensor per layer (independent allocations -> per-layer host attributes).
    sources: dict[str, list[torch.Tensor]]
    # marlin/b12x per-expert global scales ([L*E]); None for formats without them
    gate_up_alpha: torch.Tensor | None = field(default=None)
    down_alpha: torch.Tensor | None = field(default=None)
    # ModelOpt activation globals, flat [L*E] float32 sidecars when complete.
    gate_up_input_scale: torch.Tensor | None = field(default=None)
    down_input_scale: torch.Tensor | None = field(default=None)
    # Physical b12x bank interpretation. None covers native and legacy formats.
    activation_dtype: str | None = field(default=None)
    activation_dtype_reason: str | None = field(default=None)
    # per-layer HostResidency values actually applied by the loader; None -> all pinned (also the degrade signal when a request was not honored)
    layer_residency: list[str] | None = field(default=None)
    # Legacy compatibility surface. HOT rows now stay authoritative in ``sources``
    # and are streamed into protected GPU slots through bounded staging.
    hot_sources: dict[str, list[torch.Tensor | None]] = field(default_factory=dict)
    hot_expert_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)
    # Fixed row capacity of each compact HOT layer. This may exceed the seeded
    # expert count so online adaptation can start all-cold and fill rows later.
    hot_expert_capacity: dict[int, int] = field(default_factory=dict)
    # True iff the ``layer_sink`` passed to the loader was actually engaged (each layer
    # streamed straight to its sink instead of staying materialized here) -- set by
    # convert.py's per-format streaming gate; ``sources`` may hold released tensors.
    streamed: bool = False


_PARALLEL_CHUNK = 8 << 20  # default O_DIRECT chunk for the parallel reader


def _resolve_nvfp4_gpu_policy(
    model_config,
    device: torch.device,
    *,
    has_input_scales: bool,
    input_scale_unusable_reason: str | None = None,
) -> tuple[str, str, str]:
    """Resolve backend and activation layout after checkpoint sidecars are known."""
    from freetoken.moe.nvfp4_backends import (
        _b12x_a4_unusable_reason,
        resolve_moe_activation_dtype,
        select_nvfp4_backend,
    )

    cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    activation_request = getattr(model_config, "moe_activation_dtype", "auto")
    a4_reason = (
        _b12x_a4_unusable_reason()
        if cc == (12, 0)
        and has_input_scales
        and (activation_request or "auto").strip().lower() != "bf16"
        else None
    )
    candidate_dtype, _ = resolve_moe_activation_dtype(
        activation_request,
        compute_capability=cc,
        has_input_scales=has_input_scales,
        input_scale_unusable_reason=input_scale_unusable_reason,
        b12x_a4_reason=a4_reason,
    )

    configured_backend = getattr(model_config, "nvfp4_backend", None)
    implicit_backend = configured_backend is None
    if implicit_backend:
        backend_request = "auto" if candidate_dtype == "nvfp4" else "triton"
    else:
        backend_request = configured_backend
    prefer_b12x_a4 = candidate_dtype == "nvfp4" and (
        implicit_backend or backend_request == "auto"
    )
    backend = select_nvfp4_backend(
        device,
        getattr(model_config, "moe_intermediate_size", None),
        backend_request,
        activation=(
            getattr(model_config, "moe_activation", None)
            or getattr(model_config, "hidden_act", "silu")
        ),
        prefer_b12x_a4=prefer_b12x_a4,
    )
    selected_reason = (
        a4_reason
        if backend == "b12x"
        else f"selected expert backend {backend!r} is not flashinfer b12x"
    )
    activation_dtype, activation_reason = resolve_moe_activation_dtype(
        activation_request,
        compute_capability=cc,
        has_input_scales=has_input_scales,
        input_scale_unusable_reason=input_scale_unusable_reason,
        b12x_a4_reason=selected_reason,
    )
    return backend, activation_dtype, activation_reason


def _nvfp4_load_activation_dtype(
    model_config, device: torch.device, decode_target: str
) -> str:
    """Resolve whether activation sidecars can affect this load.

    Only sm_120 GPU decode can use NVFP4 activations. Returning BF16 here lets the
    source loader skip its separate load-time pass over the checkpoint shards. Final
    policy resolution still runs after loading and validates explicit requests.
    """
    requested = getattr(model_config, "moe_activation_dtype", "auto")
    requested = (requested or "auto").strip().lower()
    if requested == "bf16" or decode_target == "cpu" or device.type != "cuda":
        return "bf16"
    if torch.cuda.get_device_capability(device) != (12, 0):
        return "bf16"
    return requested


def _v4_unsupported(quant):
    raise NotImplementedError(
        f"parallel expert reader not implemented for quant {quant!r} yet; "
        "add a load_*_expert_sources_parallel using freetoken.models.weight."
        "iter_expert_tensors_parallel (only ds_fp4 implemented so far)"
    )


def _bf16_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    from freetoken.models.weight import load_moe_expert_sources

    # bf16 banks are written as-loaded -> streamable (dummy fabricates in one shot, so a
    # sink given alongside dummy=True would never fire; keep it materialize-only there).
    sink = None if dummy else layer_sink
    gate_up_source, down_source = load_moe_expert_sources(
        model_path, dtype=dtype, dummy=dummy, parallel=parallel, workers=workers, chunk=chunk,
        layer_sink=sink,
    )
    return ExpertBanks(
        "bf16", {"gate_up": gate_up_source, "down": down_source}, streamed=sink is not None
    )


class _RepackedBank:
    """A repacked per-layer bank forwarded to the converter's outer sink: the post-repack
    tensor (a reinterpreted view of the native source's storage) plus a ``release`` that
    frees that native source. Exposes just the ``.tensor``/``.nbytes``/``.release()`` the
    :class:`~freetoken.checkpoint.convert._ConvertSink` reads."""

    __slots__ = ("tensor", "nbytes", "_src")

    def __init__(self, tensor: torch.Tensor, src) -> None:
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()
        self._src = src  # the native HostBank whose storage `tensor` reinterprets

    def release(self) -> None:
        self._src.release()


class _Nvfp4RepackSink:
    """Streaming layer sink that resolves layout after activation sidecars are loaded.

    Native Triton banks pass through unchanged. Marlin and b12x banks are repacked in
    place before being forwarded to the outer FTW sink.

    Runs from the loader's reader threads and calls CUDA ops (the repack), so the whole
    per-layer body -- ``set_device`` + repack + forward + stash -- is serialized under one
    lock (conversion is disk/compute-bound, not concurrency-bound). The two native
    ``*_global`` banks fold into the alphas, so they are released here (the outer sink only
    sees, and releases, the 4 weight banks)."""

    def __init__(self, config, device: torch.device, outer) -> None:
        from freetoken.moe.nvfp4_backends import _POST_NVFP4_BANKS

        self._config = config
        self._device = device
        self._outer = outer
        self.backend: str | None = None
        self.activation_dtype = "bf16"
        self.activation_dtype_reason = "activation sidecars have not been loaded"
        self._post_names = _POST_NVFP4_BANKS
        self._lock = threading.Lock()
        self._gate_up_alpha: dict[int, torch.Tensor] = {}
        self._down_alpha: dict[int, torch.Tensor] = {}
        # post-repack per-layer views, kept only to reassemble ExpertBanks.sources (they
        # alias storage the outer sink released after writing -- released-tensor caveat).
        self._post: dict[str, dict[int, torch.Tensor]] = {n: {} for n in self._post_names}

    def set_nvfp4_input_scales(
        self,
        sources: dict[str, list[torch.Tensor]],
        *,
        input_scale_unusable_reason: str | None = None,
    ) -> None:
        has_scales = all(
            name in sources for name in ("gate_up_input_scale", "down_input_scale")
        )
        (
            self.backend,
            self.activation_dtype,
            self.activation_dtype_reason,
        ) = _resolve_nvfp4_gpu_policy(
            self._config,
            self._device,
            has_input_scales=has_scales,
            input_scale_unusable_reason=input_scale_unusable_reason,
        )

    def __call__(self, layer_id: int, banks: dict) -> None:
        from freetoken.moe.nvfp4_backends import _NATIVE_NVFP4_BANKS

        with self._lock:
            if self.backend is None:
                raise RuntimeError(
                    "NVFP4 activation sidecars must be resolved before expert layers"
                )
            if self.backend == "triton":
                self._outer(layer_id, banks)
                return
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            from freetoken.moe.nvfp4_backends import (
                b12x_repack_layer,
                marlin_repack_layer,
            )

            repack_layer = {
                "marlin": marlin_repack_layer,
                "b12x": b12x_repack_layer,
            }[self.backend]
            layer_tensors = {name: banks[name].tensor for name in _NATIVE_NVFP4_BANKS}
            post, gate_up_alpha, down_alpha = repack_layer(
                layer_tensors, self._config, self._device,
                **(
                    {"activation_dtype": self.activation_dtype}
                    if self.backend == "b12x" else {}
                ),
            )
            self._gate_up_alpha[layer_id] = gate_up_alpha
            self._down_alpha[layer_id] = down_alpha
            forwarded = {}
            for name in self._post_names:
                self._post[name][layer_id] = post[name]
                # post[name] reinterprets banks[name]'s storage in place; release it via that bank.
                forwarded[name] = _RepackedBank(post[name], banks[name])
            # the two *_global banks were consumed into the alphas; they are not forwarded,
            # so release them here instead of leaving them resident.
            banks["gate_up_global"].release()
            banks["down_global"].release()
            self._outer(layer_id, forwarded)

    def assemble(self, num_layers: int):
        """After the load: flat layer-major ``[L*E]`` alphas + the 4-name post-repack
        per-layer source lists. Asserts every layer streamed through."""
        assert self.backend in ("marlin", "b12x")
        for by_layer, what in (
            (self._gate_up_alpha, "gate_up_alpha"),
            (self._down_alpha, "down_alpha"),
        ):
            missing = [l for l in range(num_layers) if l not in by_layer]
            assert not missing, f"nvfp4 repack sink never saw layers {missing} ({what})"
        gate_up_alpha = torch.cat([self._gate_up_alpha[l] for l in range(num_layers)])
        down_alpha = torch.cat([self._down_alpha[l] for l in range(num_layers)])
        sources = {
            name: [self._post[name][l] for l in range(num_layers)] for name in self._post_names
        }
        return sources, gate_up_alpha, down_alpha


def _nvfp4_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    from freetoken.models.weight import load_nvfp4_moe_expert_sources
    from freetoken.moe.nvfp4_backends import (
        b12x_repack_sources_inplace,
        marlin_repack_sources_inplace,
        resolve_moe_activation_dtype,
    )

    # Source checkpoints expose activation sidecars immediately before their first
    # layer. A streaming conversion therefore resolves its physical layout in this
    # wrapper, while a materialized serving load resolves after the loader returns.
    repack_sink = None
    if decode_target == "gpu" and not dummy and layer_sink is not None:
        repack_sink = _Nvfp4RepackSink(model_config, device, layer_sink)
        sink = repack_sink
    else:
        sink = None if dummy else layer_sink

    load_config = model_config
    load_activation_dtype = _nvfp4_load_activation_dtype(
        model_config, device, decode_target
    )
    if load_activation_dtype == "bf16" and (
        getattr(model_config, "moe_activation_dtype", "auto") != "bf16"
    ):
        load_config = copy.copy(model_config)
        object.__setattr__(load_config, "moe_activation_dtype", "bf16")

    sources = load_nvfp4_moe_expert_sources(
        model_path, load_config, dummy=dummy, parallel=parallel, workers=workers, chunk=chunk,
        layer_sink=sink,
    )
    gate_up_input_scale = (
        torch.cat(sources["gate_up_input_scale"])
        if "gate_up_input_scale" in sources else None
    )
    down_input_scale = (
        torch.cat(sources["down_input_scale"])
        if "down_input_scale" in sources else None
    )
    has_input_scales = gate_up_input_scale is not None and down_input_scale is not None
    input_scale_unusable_reason = getattr(
        sources, "input_scale_unusable_reason", None
    )
    cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    if decode_target == "cpu":
        backend = "triton"
        activation_dtype, activation_reason = resolve_moe_activation_dtype(
            getattr(model_config, "moe_activation_dtype", "auto"),
            compute_capability=cc,
            has_input_scales=has_input_scales,
            input_scale_unusable_reason=input_scale_unusable_reason,
            b12x_a4_reason="CPU decode uses the native Triton expert layout",
        )
    elif repack_sink is not None:
        if repack_sink.backend is None:
            raise RuntimeError("NVFP4 loader did not report activation sidecars")
        backend = repack_sink.backend
        activation_dtype = repack_sink.activation_dtype
        activation_reason = repack_sink.activation_dtype_reason
    else:
        backend, activation_dtype, activation_reason = _resolve_nvfp4_gpu_policy(
            model_config,
            device,
            has_input_scales=has_input_scales,
            input_scale_unusable_reason=input_scale_unusable_reason,
        )
    # CPU-compute decode (cpu/hybrid) reads the native ModelOpt rows directly (its
    # dequant-in-GEMV kernel), so keep the native "nvfp4" layout and skip the GPU-tiled
    # marlin/b12x repacks (which only the GPU W4A16 kernels can read).
    if decode_target == "cpu":
        return ExpertBanks("nvfp4", {name: sources[name] for name in _BANK_SCHEMAS["nvfp4"]},
                           gate_up_input_scale=gate_up_input_scale,
                           down_input_scale=down_input_scale,
                           activation_dtype=activation_dtype,
                           activation_dtype_reason=activation_reason,
                           streamed=sink is not None)
    # Pick the expert-GEMM backend by compute capability (and MoE width: auto keeps
    # small-I MoE on the Triton M=1 GEMV, which beats b12x's tensor cores at single-stream
    # decode) and repack the banks (in place; the tiled blocks are byte-identical per
    # expert) into that backend's layout. One layout per process: prefill and decode read it.
    logger.info(f"NVFP4 expert backend: {backend}")
    if backend == "triton":
        return ExpertBanks("nvfp4", {name: sources[name] for name in _BANK_SCHEMAS["nvfp4"]},
                           gate_up_input_scale=gate_up_input_scale,
                           down_input_scale=down_input_scale,
                           activation_dtype=activation_dtype,
                           activation_dtype_reason=activation_reason,
                           streamed=sink is not None)
    quant_format = f"nvfp4_{backend}"
    if repack_sink is not None:
        # Streamed conversion: each layer was already repacked + written by the wrapper.
        num_layers = len(sources["gate_up_packed"])
        post_sources, gate_up_alpha, down_alpha = repack_sink.assemble(num_layers)
        return ExpertBanks(
            quant_format, post_sources,
            gate_up_alpha=gate_up_alpha, down_alpha=down_alpha,
            gate_up_input_scale=gate_up_input_scale,
            down_input_scale=down_input_scale,
            activation_dtype=activation_dtype,
            activation_dtype_reason=activation_reason,
            streamed=True,
        )
    repack = {"marlin": marlin_repack_sources_inplace, "b12x": b12x_repack_sources_inplace}
    repack_kw = {"activation_dtype": activation_dtype} if backend == "b12x" else {}
    sources = repack[backend](sources, model_config, device, **repack_kw)
    return ExpertBanks(
        quant_format,
        {name: sources[name] for name in _BANK_SCHEMAS[quant_format]},
        gate_up_alpha=sources["gate_up_alpha"],
        down_alpha=sources["down_alpha"],
        gate_up_input_scale=gate_up_input_scale,
        down_input_scale=down_input_scale,
        activation_dtype=activation_dtype,
        activation_dtype_reason=activation_reason,
    )


def _q4_0_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    if parallel:
        raise NotImplementedError(
            "parallel reader not implemented for q4_0: GGUF is a single packed file "
            "(not safetensors), so the common reader doesn't apply -- it needs a GGUF-native "
            "parallel reader (parse the tensor table, chunked O_DIRECT over the one file)"
        )
    from freetoken.models.weight import load_q4_0_moe_expert_sources

    # Native GGUF Q4_0 routed experts: packed block bytes streamed to the GPU and
    # dequantized inside the borrowed ggml MoE kernels (no bf16 expert copy). Banks are
    # per-layer HostBanks (pin-after-fill), so conversion streams each completed layer's
    # gate_up + down straight through the sink (dummy fabricates in one shot -> not streamed).
    sink = None if dummy else layer_sink
    sources = load_q4_0_moe_expert_sources(model_path, model_config, dummy=dummy, layer_sink=sink)
    return ExpertBanks(
        "q4_0", {name: sources[name] for name in _BANK_SCHEMAS["q4_0"]}, streamed=sink is not None
    )


def _dsfp4_banks(model_path, model_config, device, dtype, dummy, parallel=False, workers=8, chunk=_PARALLEL_CHUNK, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    args = model_config.dsv4_args
    assert args is not None, "ds_fp4 expert banks require dsv4_args on the model config"
    # DeepSeek-FP4: packed e2m1 + e8m0 per-32 block scales, no global scale -> 4 banks,
    # no alphas. DeepSeek-V4's own grouped GEMV kernels read them via bank_views().
    # Written as-loaded -> streamable (dummy fabricates in one shot; never streamed).
    sink = None if dummy else layer_sink
    if dummy:
        from freetoken.models.deepseek_v4.weight import dummy_dsfp4_expert_sources

        banks = dummy_dsfp4_expert_sources(args)
    elif parallel:  # parallel: common chunked multi-threaded O_DIRECT reader
        from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources_parallel

        banks = load_dsfp4_expert_sources_parallel(
            model_path, args, workers=workers, chunk=chunk, layer_sink=sink
        )
    else:
        from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources

        banks = load_dsfp4_expert_sources(model_path, args, layer_sink=sink)
    return ExpertBanks(
        "ds_fp4", {name: banks[name] for name in _BANK_SCHEMAS["ds_fp4"]}, streamed=sink is not None
    )


def _model_setup_override(model_config):
    architectures = getattr(model_config, "architectures", None)
    if not architectures:
        return None

    from freetoken.models.register import _load_attr, get_model_spec

    try:
        spec = get_model_spec(architectures[0])
    except ValueError:
        return None
    try:
        return _load_attr(spec.module, "setup_offload_expert_banks")
    except AttributeError:
        return None


# ModelConfig.expert_quant -> provider
_PROVIDERS = {
    "none": _bf16_banks,
    "nvfp4": _nvfp4_banks,
    "ds_fp4": _dsfp4_banks,
    "q4_0": _q4_0_banks,
}


def _build_expert_banks(model_path, model_config, device, dtype, dummy, parallel, workers, chunk, decode_target="gpu", layer_sink=None) -> ExpertBanks:
    """Dispatch to the model's setup-override or the per-quant provider. ``parallel=True``
    is the parallel read; a provider that hasn't implemented it raises NotImplementedError (the
    caller falls back to serial). ``decode_target`` lets the cpu backend force CPU-readable
    (native, non-GPU-tiled) bank layouts. ``layer_sink`` (converter only) is forwarded to
    setups/providers that declare the parameter; the rest ignore it and stay on the
    materialize-and-write path (``ExpertBanks.streamed`` reports which happened)."""
    setup = _model_setup_override(model_config)
    if setup is not None:
        import inspect

        params = inspect.signature(setup).parameters
        supports_parallel = "parallel" in params
        if parallel and not supports_parallel:
            arch = getattr(model_config, "architectures", ["?"])[0]
            raise NotImplementedError(
                f"parallel reader not implemented for {arch} (model owns expert setup "
                "via setup_offload_expert_banks; add a parallel path there)"
            )
        kw = dict(device=device, dtype=dtype, dummy=dummy)
        if supports_parallel:
            kw.update(parallel=parallel, workers=workers, chunk=chunk)
        if "decode_target" in params:
            kw["decode_target"] = decode_target
        if "layer_sink" in params and layer_sink is not None:
            kw["layer_sink"] = layer_sink
        return setup(model_path, model_config, **kw)

    expert_quant = model_config.expert_quant
    if expert_quant not in _PROVIDERS:
        raise ValueError(
            f"no expert-bank provider for expert_quant={expert_quant!r} "
            f"(known: {sorted(_PROVIDERS)})"
        )
    return _PROVIDERS[expert_quant](
        model_path, model_config, device, dtype, dummy,
        parallel=parallel, workers=workers, chunk=chunk, decode_target=decode_target,
        layer_sink=layer_sink,
    )


def _host_ram_fits_parallel(model_path: str) -> bool:
    """Best-effort: can free host RAM hold the expert banks plus the parallel reader's one
    extra (non-reclaimable) whole-shard buffer? Unknown (non-local path / no /proc) -> True,
    i.e. keep the fast path. Banks ~= checkpoint size (experts dominate); transient ~= the
    largest shard. Uses MemAvailable (counts reclaimable cache) -- the OOM-relevant figure."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:
        return True
    try:  # resolve a hub id to its local cache dir (no-op for a local path) so glob sees the shards
        from freetoken.utils.hf import download_hf_weight

        model_path = download_hf_weight(model_path)
    except Exception:
        return True
    sizes = [os.path.getsize(p) for p in glob.glob(os.path.join(model_path, "*.safetensors"))]
    if not sizes:
        return True
    return avail > sum(sizes) + max(sizes)


def ftw_bank_bytes(model_path: str) -> int | None:
    """Total expert-bank bytes of an FTW checkpoint, from its metadata (no bank IO).
    ``None`` when the checkpoint is not FTW -- callers that size things pre-load (auto split residency) then leave the load unchanged."""
    import json

    meta = os.path.join(model_path, "freetoken_weight.json")
    if not os.path.isfile(meta):
        return None
    with open(meta, encoding="utf-8") as f:
        tensors = json.load(f).get("tensors", [])
    return sum(t["nbytes"] for t in tensors if t.get("kind") == "experts_bank")


def ftw_bank_byte_breakdown(model_path: str) -> tuple[int, int] | None:
    """Return ``(row_bytes, fixed_bytes)`` for an FTW's expert-bank entries.

    Row bytes are the per-expert tensors whose residency can be split. Fixed bytes
    are the small alpha vectors that remain pinned regardless of layer residency.
    ``None`` means the checkpoint is not FTW.
    """
    import json

    meta = os.path.join(model_path, "freetoken_weight.json")
    if not os.path.isfile(meta):
        return None
    with open(meta, encoding="utf-8") as f:
        tensors = json.load(f).get("tensors", [])
    row_bytes = fixed_bytes = 0
    for tensor in tensors:
        if tensor.get("kind") != "experts_bank":
            continue
        base = str(tensor.get("name", "")).split("#L", 1)[0]
        if base in (
            "gate_up_alpha", "down_alpha",
            "gate_up_input_scale", "down_input_scale",
        ):
            fixed_bytes += int(tensor["nbytes"])
        else:
            row_bytes += int(tensor["nbytes"])
    return row_bytes, fixed_bytes


def bank_bytes_estimate(model_config) -> int | None:
    """Estimated total expert-bank bytes of a raw checkpoint, from the model config alone.

    Sizes the pin-budget decisions where FTW metadata is not available; ``None`` for unknown formats or missing dims (callers then skip the pre-load sizing).
    nvfp4 uses the native-row formula, a slight over-estimate for the repacked backends."""
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (
        getattr(model_config, "moe_weight_format", None) or "bf16"
    )
    per_expert = _BANK_BYTES_PER_EXPERT.get(fmt)
    layers = getattr(model_config, "num_moe_layers", None)
    experts = getattr(model_config, "num_experts", None)
    hidden = getattr(model_config, "hidden_size", None)
    inter = getattr(model_config, "moe_intermediate_size", None)
    if per_expert is None or not all((layers, experts, hidden, inter)):
        return None
    return layers * experts * per_expert(hidden, inter)


def bank_bytes_per_expert(model_config) -> int | None:
    """Logical row-bank bytes for one expert in one layer."""
    total = bank_bytes_estimate(model_config)
    layers = getattr(model_config, "num_moe_layers", None)
    experts = getattr(model_config, "num_experts", None)
    if total is None or not layers or not experts:
        return None
    return total // (layers * experts)


def _load_expert_banks_impl(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool | None = None,
    workers: int = 8,
    chunk: int = _PARALLEL_CHUNK,
    decode_target: str = "gpu",
    layer_sink=None,
    layer_residency: list[str] | None = None,
    disk_pager=None,
    hot_expert_ids: dict[int, tuple[int, ...]] | None = None,
    hot_expert_capacity: dict[int, int] | None = None,
    bank_source: str = "auto",
    hugepages_tmpfs: str | None = None,
    hugepages_tmpfs_margin_bytes: int = 1 << 30,
) -> ExpertBanks:
    """Load (or fabricate, with ``dummy=True``) the expert banks. Two paths, both returning
    the same normalized ``ExpertBanks`` and both pinning after fill:

    * **FTW path**: if ``model_path`` is a converted FTW checkpoint, read its
      repacked banks directly (contiguous chunked O_DIRECT). No auto-conversion.
    * **Indexed path**: for a byte-identical packed safetensors layout, map DISK layers
      directly from their source shards and copy resident layers from the same ranges.
    * **Slow path** (the original checkpoint): auto-pick **parallel** (the common parallel chunked
      O_DIRECT reader) when experts are stored as many small tensors -- the serial read is
      slow there -- else the **serial baseline** (packed experts: serial already saturates,
      parallel only adds read amplification). parallel unavailable for a quant falls back to serial.

    ``parallel`` overrides the slow-path auto-pick: ``None`` = auto (production), ``True`` /
    ``False`` = force parallel / serial (used by the loader benchmark and the converter).

    ``layer_sink`` (the converter only): forwarded to whichever provider is picked; a
    provider only engages it (and reports ``ExpertBanks.streamed=True``) for its own
    streamable formats, so callers must check ``streamed`` rather than assume it fired.

    ``layer_residency``: per-layer ``HostResidency`` labels applied at settle time -- explicitly on the FTW fast path, ambiently (``requested_residency``) in the slow-path providers.
    Applied labels are echoed on ``ExpertBanks.layer_residency``; a loader that settles some other way leaves it ``None`` (CPU-layer decode still works on pinned banks, it just saves no pin quota).

    ``disk_pager`` selects anonymous UFFD regions for FTW or indexed DISK layers.
    ``None`` keeps the portable file-mapping backend.
    """
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.checkpoint.safetensors_bank_index import load_indexed_banks
    from freetoken.moe.host_banks import HostResidency

    resolved_source = "source" if dummy else resolve_bank_source(model_path, bank_source)
    ftw = resolved_source == "ftw"
    indexed = resolved_source == "index"

    wants_disk = bool(
        layer_residency
        and HostResidency.DISK.value in layer_residency
    )
    if hugepages_tmpfs and not wants_disk:
        raise ValueError(
            "--moe-bank-hugepages-tmpfs requires at least one DISK expert layer"
        )
    if hugepages_tmpfs and disk_pager is not None:
        raise ValueError(
            "--moe-bank-hugepages-tmpfs cannot be used with --moe-disk-pager uffd"
        )
    if (wants_disk or hot_expert_ids or hot_expert_capacity) and (
        not model_path or not (ftw or indexed)
    ):
        raise ValueError(
            "DISK and HOT expert-bank residency require FTW or a byte-identical "
            "safetensors bank index; convert repacked layouts with `ft checkpoint`"
        )

    if ftw and not dummy:
        banks = load_ftw_banks(
            model_path, num_layers=model_config.num_moe_layers, workers=workers, chunk=chunk,
            layer_residency=layer_residency, disk_pager=disk_pager,
            hot_expert_ids=hot_expert_ids,
            hot_expert_capacity=hot_expert_capacity,
            hugepages_tmpfs=hugepages_tmpfs,
            hugepages_tmpfs_margin_bytes=hugepages_tmpfs_margin_bytes,
        )
        if banks is not None:
            logger.info_rank0(f"expert banks: FTW fast path (FTW checkpoint {model_path})")
            if banks.quant_format == "nvfp4" and decode_target == "gpu":
                from freetoken.moe.nvfp4_backends import (
                    b12x_repack_sources_inplace,
                )

                has_input_scales = (
                    banks.gate_up_input_scale is not None
                    and banks.down_input_scale is not None
                )
                backend, activation_dtype, activation_reason = (
                    _resolve_nvfp4_gpu_policy(
                        model_config,
                        device,
                        has_input_scales=has_input_scales,
                        input_scale_unusable_reason=banks.activation_dtype_reason,
                    )
                )
                if activation_dtype == "nvfp4":
                    assert backend == "b12x"
                    copied_sources = {
                        name: [
                            torch.empty_like(
                                tensor, pin_memory=tensor.is_pinned()
                            ).copy_(tensor)
                            for tensor in rows
                        ]
                        for name, rows in banks.sources.items()
                    }
                    packed = b12x_repack_sources_inplace(
                        copied_sources, model_config, device,
                        activation_dtype=activation_dtype,
                    )
                    banks = replace(
                        banks,
                        quant_format="nvfp4_b12x",
                        sources={
                            name: packed[name]
                            for name in _BANK_SCHEMAS["nvfp4_b12x"]
                        },
                        gate_up_alpha=packed["gate_up_alpha"],
                        down_alpha=packed["down_alpha"],
                        activation_dtype=activation_dtype,
                        activation_dtype_reason=activation_reason,
                    )
                elif backend == "triton":
                    banks = replace(
                        banks,
                        activation_dtype=activation_dtype,
                        activation_dtype_reason=activation_reason,
                    )
                else:
                    raise RuntimeError(
                        f"native NVFP4 FTW banks cannot use selected {backend!r} "
                        "W4A16 backend without an already-repacked layout; reconvert "
                        "the FTW with that explicit backend"
                    )
            return banks

    if indexed and not dummy:
        banks = load_indexed_banks(
            model_path,
            num_layers=model_config.num_moe_layers,
            dtype=dtype,
            layer_residency=layer_residency,
            disk_pager=disk_pager,
            hot_expert_ids=hot_expert_ids,
            hot_expert_capacity=hot_expert_capacity,
            hugepages_tmpfs=hugepages_tmpfs,
            hugepages_tmpfs_margin_bytes=hugepages_tmpfs_margin_bytes,
        )
        logger.info_rank0(
            f"expert banks: safetensors indexed path (checkpoint {model_path})"
        )
        return banks

    if parallel and not _PARALLEL_READER_SUPPORTED:
        logger.warning_rank0(
            "expert banks: parallel O_DIRECT reader unsupported on this platform "
            "(no os.O_DIRECT/preadv) -> serial build"
        )
        parallel = False

    auto = parallel is None
    if auto:
        from freetoken.models.weight import experts_scattered

        parallel = _PARALLEL_READER_SUPPORTED and not dummy and experts_scattered(model_path)
        # Low-RAM fallback: the parallel reader holds whole-shard ANONYMOUS buffers
        # (non-reclaimable) on top of the ~bank-sized resident set, so on a memory-tight box
        # it OOMs where the serial path (reclaimable file mmap) survives. Drop to serial when
        # free RAM can't cover the banks + one shard's transient. (--expert-load serial/parallel
        # bypass this by forcing ``parallel`` explicitly.)
        if parallel and not _host_ram_fits_parallel(model_path):
            logger.warning_rank0(
                "expert banks: low free RAM -> serial build (avoids parallel-reader OOM; "
                "override with --expert-load parallel)"
            )
            parallel = False
    logger.info_rank0(f"expert banks: slow path ({'parallel' if parallel else 'serial'} build)")
    # parallel's reader resolves hub ids + handles single-file/no-index checkpoints, so it won't
    # OSError on those (which would leak the banks it pre-allocated, since host banks live for
    # the process). Only NotImplementedError (quant has no parallel reader; raised before any
    # allocation) falls back to serial.
    from freetoken.moe.host_banks import requested_residency

    with requested_residency(layer_residency) as residency_plan:
        try:
            banks = _build_expert_banks(model_path, model_config, device, dtype, dummy, parallel, workers, chunk,
                                        decode_target, layer_sink)
        except NotImplementedError as exc:
            if not parallel:
                raise
            logger.warning_rank0(f"parallel reader unavailable ({exc}); falling back to serial build")
            banks = _build_expert_banks(model_path, model_config, device, dtype, dummy, False, workers, chunk,
                                        decode_target, layer_sink)
    return _echo_residency(banks, layer_residency, residency_plan)


def load_expert_banks(
    model_path: str,
    model_config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool | None = None,
    workers: int = 8,
    chunk: int = _PARALLEL_CHUNK,
    decode_target: str = "gpu",
    layer_sink=None,
    layer_residency: list[str] | None = None,
    disk_pager=None,
    hot_expert_ids: dict[int, tuple[int, ...]] | None = None,
    hot_expert_capacity: dict[int, int] | None = None,
    bank_source: str = "auto",
    hugepages: str = "auto",
    hugepages_tmpfs: str | None = None,
    hugepages_tmpfs_margin_gib: float = 1.0,
) -> ExpertBanks:
    """Load expert banks under one THP policy and emit its startup probe report."""
    from freetoken.moe.host_banks import (
        current_hugepage_scope,
        format_hugepage_status,
        read_meminfo_hugepages,
        requested_hugepages,
    )

    margin_gib = float(hugepages_tmpfs_margin_gib)
    if not math.isfinite(margin_gib) or margin_gib < 0:
        raise ValueError("tmpfs hugepage margin must be finite and non-negative")
    if hugepages == "off" and hugepages_tmpfs:
        raise ValueError(
            "--moe-bank-hugepages-tmpfs requires --moe-bank-hugepages auto or on"
        )
    margin_bytes = int(margin_gib * (1 << 30))
    active_scope = current_hugepage_scope()
    if active_scope is not None:
        if active_scope.mode != hugepages:
            raise RuntimeError(
                f"expert-bank hugepage policy {hugepages!r} conflicts with active "
                f"model-load policy {active_scope.mode!r}"
            )
        banks = _load_expert_banks_impl(
            model_path,
            model_config,
            device=device,
            dtype=dtype,
            dummy=dummy,
            parallel=parallel,
            workers=workers,
            chunk=chunk,
            decode_target=decode_target,
            layer_sink=layer_sink,
            layer_residency=layer_residency,
            disk_pager=disk_pager,
            hot_expert_ids=hot_expert_ids,
            hot_expert_capacity=hot_expert_capacity,
            bank_source=bank_source,
            hugepages_tmpfs=hugepages_tmpfs,
            hugepages_tmpfs_margin_bytes=margin_bytes,
        )
        active_scope.sources.update(banks.sources)
        return banks
    before = read_meminfo_hugepages()
    with requested_hugepages(hugepages) as scope:
        banks = _load_expert_banks_impl(
            model_path,
            model_config,
            device=device,
            dtype=dtype,
            dummy=dummy,
            parallel=parallel,
            workers=workers,
            chunk=chunk,
            decode_target=decode_target,
            layer_sink=layer_sink,
            layer_residency=layer_residency,
            disk_pager=disk_pager,
            hot_expert_ids=hot_expert_ids,
            hot_expert_capacity=hot_expert_capacity,
            bank_source=bank_source,
            hugepages_tmpfs=hugepages_tmpfs,
            hugepages_tmpfs_margin_bytes=margin_bytes,
        )
        scope.sources.update(banks.sources)
    after = read_meminfo_hugepages()
    for line in format_hugepage_status(scope, hugepages, before, after).splitlines():
        logger.info_rank0(line)
    return banks


def _echo_residency(banks: ExpertBanks, requested, plan) -> ExpertBanks:
    """Stamp an honored residency request onto the ExpertBanks; keep None (and warn) when no settle point consulted the plan."""
    if requested is None or banks.layer_residency is not None:
        return banks
    if plan is not None and plan.applied:
        import dataclasses

        labels = [plan.actual.get(i, r) for i, r in enumerate(requested)]
        downgraded = [i for i, r in enumerate(requested) if labels[i] != r]
        if downgraded:
            logger.warning_rank0(
                f"--moe-cpu-layers: layers {downgraded} settled pageable instead of "
                f"OS-locked (lock failed); they still decode on the CPU executor but "
                f"may swap under memory pressure"
            )
        return dataclasses.replace(banks, layer_residency=labels)
    from freetoken.moe.host_banks import HostResidency

    if any(r != HostResidency.PINNED.value for r in requested):
        logger.warning_rank0(
            "--moe-cpu-layers: this checkpoint's bank loader settles banks without "
            "per-layer residency (pre-pins everything); CPU-layer decode still works "
            "but saves no pinned quota"
        )
    return banks
