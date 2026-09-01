"""Convert an HF safetensors checkpoint into a FreeToken Weight (FTW) checkpoint.

Model-agnostic: it drives the *existing* per-model loaders once and stores their output, so
no per-model conversion code is needed.

* dense weights = exactly what ``load_weight(include_moe_experts=...)`` yields (post
  fusion/TP-shard) -> ``kind="weight"``; at load they feed ``model.load_state_dict``.
* optional Qwen3.8 MTP weights = the separately gated head reader output ->
  ``kind="mtp"``; default conversions omit it.
* offload experts = exactly what ``load_expert_banks(parallel=True)`` produces (post
  backend-repack pinned banks + alpha scale vectors) -> ``kind="experts_bank"`` (alphas are
  told apart at load by their reserved names, so they need no separate kind).

The output directory is a self-contained checkpoint (config + tokenizer copied), so you can
point ``--model`` straight at it; the load path auto-detects the FTW and reads it (FTW).
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import threading

import torch

from .ftw import DEFAULT_SHARD_LIMIT, FTWWriter, layer_bank_entry_name

# Machine-readable convert progress for a supervising process (e.g. a GUI frontend parses
# these `FTCONVERT <phase> <done> <total>` stdout lines to drive its convert bar). Gated by
# FREETOKEN_CONVERT_PROGRESS=1 so plain CLI use isn't spammed; the human tqdm bars stay on
# stderr. Phases: `dense` (indeterminate, done=total=0), `experts` (byte totals), `finalize`.
_EMIT_PROGRESS = os.environ.get("FREETOKEN_CONVERT_PROGRESS") == "1"


def _progress(phase: str, done: int = 0, total: int = 0) -> None:
    if _EMIT_PROGRESS:
        print(f"FTCONVERT {phase} {done} {total}", flush=True)


def _source_fingerprint(
    model_path: str,
    model_config,
    *,
    device,
    mtp_quant: str | None = None,
) -> str:
    """Identity of (checkpoint + quant + GPU capability), stored in the FTW index so
    it's clear what an FTW was built from. Cheap (stat only)."""
    h = hashlib.sha256()
    h.update(f"quant={getattr(model_config, 'expert_quant', None)}|".encode())
    h.update(f"mtp_quant={mtp_quant}|".encode())
    h.update(f"arch={getattr(model_config, 'architectures', None)}|".encode())
    try:  # nvfp4 marlin/b12x layout depends on compute capability
        h.update(f"cc={torch.cuda.get_device_capability(device)}|".encode())
    except Exception:
        pass
    files = sorted(
        glob.glob(os.path.join(model_path, "*.safetensors")) + glob.glob(os.path.join(model_path, "*.gguf"))
    )
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:16]

# Checkpoint metadata to carry over so the FTW dir is a usable checkpoint on its own.
# (Weight shards + the safetensors index are intentionally NOT copied.)
# Everything that is NOT a weight shard is metadata we carry over verbatim. A whitelist
# misses model-specific layouts (e.g. DSV4's inference/config.json + encoding/ live in
# subdirs), so we copy every non-weight file preserving its relative path instead.
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ftw")  # .ftw: a nested FTW, not source
_SKIP_NAMES = ("model.safetensors.index.json",)  # indexes shards the FTW replaces
# .freetoken_expert_cache: the legacy per-bank cache (can be tens of GB of stale .bin)
_SKIP_DIRS = (".git", ".cache", ".freetoken_expert_cache")


def _copy_metadata(model_path: str, out_dir: str) -> list[str]:
    """Copy all non-weight files (config, tokenizer, remote-code, nested model configs)
    preserving directory structure, so the FTW dir is a self-contained checkpoint."""
    if os.path.isfile(model_path):
        # A single-file source has no sibling metadata to walk: a .gguf carries its config
        # AND tokenizer in its own KV section. Emit a metadata-only copy (header + KV, no
        # weight data) the FTW dir resolves those from. We deliberately do NOT sweep the
        # file's parent dir -- an HF gguf snapshot dir can hold unrelated blobs.
        from freetoken.models.gguf.reader import (
            FTW_METADATA_GGUF,
            is_gguf_path,
            write_metadata_gguf,
        )

        if is_gguf_path(model_path):
            os.makedirs(out_dir, exist_ok=True)
            write_metadata_gguf(model_path, os.path.join(out_dir, FTW_METADATA_GGUF))
            return [FTW_METADATA_GGUF]
        return []

    out_abs = os.path.abspath(out_dir)
    copied = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and os.path.abspath(os.path.join(root, d)) != out_abs]
        for name in files:
            if name.endswith(_WEIGHT_SUFFIXES) or name in _SKIP_NAMES:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, model_path)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return copied


class _ConvertSink:
    """Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each
    completed layer's banks as their own FTW entries immediately (name
    ``f"{bank_name}#L{layer_id:05d}"``, kind ``"experts_bank"``) and releases them, so
    conversion RAM peaks at ~in-flight layers instead of the whole bank set.

    Only engaged for streamable formats -- ``ExpertBanks.streamed`` reports whether this
    actually fired; if not, the caller falls back to the materialize-and-write path
    instead. The progress bar is created lazily on the first call, so a format that never
    streams never shows one.

    ``FTWWriter`` buffers file/shard state and is not thread-safe; completion callbacks
    can fire from the loader's own reader threads, so the write+release is serialized
    under one lock (disk-bound anyway).
    """

    def __init__(self, writer: FTWWriter, desc: str = "Converting expert banks") -> None:
        self._writer = writer
        self._desc = desc
        self._bar = None
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.n_written = 0
        self.n_bytes = 0
        self.max_layer_bytes = 0

    def __call__(self, layer_id: int, banks: dict) -> None:
        with self._lock:
            assert layer_id not in self._seen, f"layer {layer_id} streamed to the sink twice"
            self._seen.add(layer_id)
            if self._bar is None:
                from freetoken.utils.progress import byte_bar

                self._bar = byte_bar(0, self._desc)  # total unknown up front (streamed)
            nbytes = 0
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    layer_bank_entry_name(bank_name, layer_id), bank.tensor, kind="experts_bank"
                )
                nbytes += bank.nbytes
                bank.release()
                self.n_written += 1
            self.n_bytes += nbytes
            self.max_layer_bytes = max(self.max_layer_bytes, nbytes)
            self._bar.update(nbytes)
            # Cumulative BYTES (not the bank count): the supervisor maps this against the
            # known expert-pool size for a smooth phase-budgeted bar. Total stays 0 (unknown
            # up front while streaming); the materialize path below emits a real total.
            _progress("experts", self.n_bytes, 0)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def num_layers(self) -> int:
        return len(self._seen)


def iter_converted_mtp_weights(weights, mtp_quant: str):
    """Yield MTP tensors in the runtime state-dict layout for one FTW precision.

    BF16 is a pass-through. NVFP4 replaces only the two routed-expert tensors with
    the six native group-16 banks used by the main experts. Attention, router, shared
    expert, hyper-connections, and all norms remain BF16.
    """
    if mtp_quant not in ("bf16", "nvfp4"):
        raise ValueError(f"--mtp-quant must be 'bf16' or 'nvfp4', got {mtp_quant!r}")
    if mtp_quant == "bf16":
        yield from weights
        return

    from freetoken.models.nvfp4_banks import quantize_nvfp4_group16

    converted = set()
    suffixes = {
        ".mlp.experts.gate_up_proj": "gate_up",
        ".mlp.experts.down_proj": "down",
    }
    for name, tensor in weights:
        role = next((r for suffix, r in suffixes.items() if name.endswith(suffix)), None)
        if role is None:
            yield name, tensor
            continue
        if role in converted:
            raise ValueError(f"MTP checkpoint contains more than one {role} expert tensor")
        converted.add(role)
        packed, scale, row_global = quantize_nvfp4_group16(tensor)
        prefix = name.rsplit(".", 1)[0]
        yield f"{prefix}.{role}_packed", packed
        yield f"{prefix}.{role}_scale", scale
        yield f"{prefix}.{role}_global", row_global
    if converted != {"gate_up", "down"}:
        raise ValueError(
            "NVFP4 MTP conversion requires gate_up_proj and down_proj routed experts; "
            f"found {sorted(converted)}"
        )


def mtp_ftw_metadata(mtp_quant: str | None, n_mtp: int, mtp_expert_bytes: int) -> dict:
    """Metadata shared by production conversion and CPU-only synthetic FTW tests."""
    if n_mtp == 0:
        return {"mtp_quant": None, "mtp_expert_bytes": 0}
    if mtp_quant not in ("bf16", "nvfp4"):
        raise ValueError(f"invalid MTP FTW quantization {mtp_quant!r}")
    return {"mtp_quant": mtp_quant, "mtp_expert_bytes": int(mtp_expert_bytes)}


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
    include_mtp: bool = False,
    mtp_quant: str = "bf16",
    nvfp4_backend: str | None = None,
) -> dict:
    """Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict.

    The FTW format is TP-agnostic and conversion runs single-process, so the resulting
    checkpoint records no TP layout and loads independently of the runtime TP setting."""
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.models.weight import load_weight
    from freetoken.moe.expert_banks import load_expert_banks
    from .ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(model_path):
        raise SystemExit(f"{model_path} is already an FTW checkpoint")
    if mtp_quant not in ("bf16", "nvfp4"):
        raise ValueError(f"--mtp-quant must be 'bf16' or 'nvfp4', got {mtp_quant!r}")
    if not include_mtp and mtp_quant != "bf16":
        raise ValueError("--mtp-quant nvfp4 requires --speculative-mtp on")
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    elif tp.size != 1:
        raise SystemExit(
            f"FTW conversion runs single-process and the format records no TP layout, "
            f"but TP is already set to size={tp.size}"
        )
    dev = torch.device(device or "cpu")
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"FTW conversion device must be CPU or CUDA, got {dev}")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        # Initialize the context before backend selection and the first per-layer repack.
        torch.zeros(1, device=dev)

    cfg = EngineConfig(model_path=model_path, tp_info=DistributedInfo(tp.rank, tp.size),
                       dtype=dtype, moe_backend=moe_backend)
    mc = cfg.model_config
    if getattr(mc, "expert_quant", None) == "nvfp4":
        # A CPU conversion keeps the checkpoint-native six-bank layout, which is portable
        # and byte-exact. A selected GPU enables the hardware-specific tiled repack. The
        # repackers only rearrange/fold source NVFP4 buffers and never dequantize weights.
        backend = nvfp4_backend or ("auto" if dev.type == "cuda" else "triton")
        object.__setattr__(mc, "nvfp4_backend", backend)
    offload = moe_backend == "offload" and getattr(mc, "is_moe", False)
    include_moe_experts = not offload

    from freetoken.utils.progress import byte_bar, count_bar

    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    n_weight = n_mtp = n_bank = n_alpha = 0
    mtp_expert_bytes = 0

    # 1) dense weights (host tensors; load straight to CPU to avoid GPU pressure)
    _progress("dense", 0, 0)  # phase start; per-tensor cumulative bytes follow (total unknown)
    dense_bytes = 0
    for name, tensor in count_bar(load_weight(model_path, torch.device("cpu"),
                                              include_moe_experts=include_moe_experts),
                                  "Converting dense weights"):
        writer.add_tensor(name, tensor, kind="weight")
        n_weight += 1
        dense_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes, 0)

    # Preserve Qwen3.8's optional MTP head under its own kind. Runtime loading still
    # ignores these tensors unless --speculative-mtp on is selected, so existing FTW
    # memory use and the default-off model state are unchanged.
    if include_mtp and getattr(mc, "qwen4_args", None) is not None:
        from freetoken.models.qwen4_exp.weight import iter_mtp_weights

        for name, tensor in count_bar(
            iter_converted_mtp_weights(
                iter_mtp_weights(model_path, torch.device("cpu")), mtp_quant
            ),
            "Converting MTP weights",
        ):
            writer.add_tensor(name, tensor, kind="mtp")
            n_mtp += 1
            if ".mlp.experts." in name:
                mtp_expert_bytes += tensor.numel() * tensor.element_size()

    # 2) offload expert banks (post-repack) + alpha scales (slow path auto-picks parallel/serial)
    quant_format = None
    num_layers = None
    max_bank_layer_bytes = 0
    expert_bank_geometry = None
    if offload:
        geometry_layers = getattr(mc, "num_moe_layers", None)
        if geometry_layers is None:
            geometry_layers = int(getattr(mc, "num_layers")) - int(
                getattr(mc, "first_k_dense_replace", 0)
            )
        expert_bank_geometry = {
            "num_layers": int(geometry_layers),
            "num_experts": int(getattr(mc, "num_experts", 0)),
            "hidden_size": int(getattr(mc, "hidden_size", 0)),
            "moe_intermediate_size": int(getattr(mc, "moe_intermediate_size", 0)),
        }
        if any(value <= 0 for value in expert_bank_geometry.values()):
            raise ValueError(f"invalid expert-bank geometry: {expert_bank_geometry}")
        # Streamable formats write each layer to its own FTW entry as it completes instead
        # of materializing the whole bank set. Native NVFP4 streams directly; marlin/b12x
        # repack one completed layer before forwarding it. Providers that cannot stream
        # ignore the sink, so the actual behavior comes from ExpertBanks.streamed.
        sink = _ConvertSink(writer)
        # Conversion favors the serial safetensors reader deliberately. Its mmap input and
        # per-layer sink keep resident memory to one expert layer, while the parallel reader
        # holds whole-shard anonymous prefetch buffers that are unsuitable for a 229 GB source.
        banks = load_expert_banks(
            model_path, mc, device=dev, dtype=dtype, parallel=False, layer_sink=sink
        )
        quant_format = banks.quant_format
        if banks.streamed:
            sink.close()
            num_layers = sink.num_layers  # however many distinct layers the sink actually saw
            n_bank = sink.n_written
            max_bank_layer_bytes = sink.max_layer_bytes
            assert num_layers > 0, (
                "provider reported streamed=True but the sink never fired -- the FTW "
                "would silently have no expert banks"
            )
            # Formats that fold their global scales (nvfp4 marlin/b12x) stream the weight
            # banks per layer but keep the alphas as flat [L*E] GPU vectors; write those as
            # flat reserved-name entries (same kind + names the materialize branch uses, so
            # the reader's reserved-name path reconstructs them identically).
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(banks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind="experts_bank")
                    n_alpha += 1
        else:
            # The on-disk format keeps one contiguous region per bank and the writer only
            # has whole-tensor add_tensor, so the per-layer sources reassemble into one
            # flat tensor (a per-bank host RAM spike during conversion).
            items = []
            for name, per_layer in banks.sources.items():
                if num_layers is None:
                    num_layers = len(per_layer)
                else:
                    assert len(per_layer) == num_layers, (name, len(per_layer), num_layers)
                items.append((name, torch.cat(per_layer, dim=0) if len(per_layer) > 1 else per_layer[0]))
            for an in ("gate_up_alpha", "down_alpha"):
                if getattr(banks, an, None) is not None:
                    items.append((an, getattr(banks, an)))
            total_bytes = sum(t.numel() * t.element_size() for _, t in items)
            max_bank_layer_bytes = sum(
                tensor.numel() * tensor.element_size() // num_layers
                for name, tensor in items
                if name not in ("gate_up_alpha", "down_alpha")
            )
            bar = byte_bar(total_bytes, "Converting expert banks")
            done_bytes = 0
            _progress("experts", 0, total_bytes)
            for name, tensor in items:
                writer.add_tensor(name, tensor, kind="experts_bank")
                nbytes = tensor.numel() * tensor.element_size()
                bar.update(nbytes)
                done_bytes += nbytes
                _progress("experts", done_bytes, total_bytes)
                n_bank += name not in ("gate_up_alpha", "down_alpha")
                n_alpha += name in ("gate_up_alpha", "down_alpha")
            bar.close()

    _progress("finalize")  # writing shard index + copying config/tokenizer
    copied = _copy_metadata(model_path, out_dir)

    try:
        fingerprint = _source_fingerprint(
            model_path, mc, device=dev, mtp_quant=mtp_quant if include_mtp else None
        )
    except Exception:
        fingerprint = None

    index = writer.finalize({
        "source_model_path": os.path.abspath(model_path),
        "fingerprint": fingerprint,
        # quant_format records the actual on-disk bank layout (e.g. nvfp4_marlin vs
        # nvfp4_b12x): the suffix is a runtime backend pick (GPU capability / env), NOT in
        # config, and the stored bytes are physically repacked into it -- so it's kept and
        # read back at load (ftw.load_ftw_banks). dtype/moe_backend were dropped: each
        # tensor already carries its own dtype, and nothing reads a model-level backend.
        "quant_format": quant_format,
        # The reader takes num_layers from the model config (copied into this
        # checkpoint); recording it here too gives load_ftw_banks a cross-check that
        # the banks match the config they ship with. None for non-offload checkpoints.
        "expert_bank_num_layers": num_layers,
        "expert_bank_geometry": expert_bank_geometry,
        # Largest completed layer handed to the writer. For streamed GLM NVFP4 this is
        # the expert-bank resident-data bound, excluding one current source tensor and
        # small scalar/global-scale bookkeeping.
        "expert_bank_max_layer_bytes": max_bank_layer_bytes,
        **mtp_ftw_metadata(mtp_quant if include_mtp else None, n_mtp, mtp_expert_bytes),
        "counts": {
            "weight": n_weight,
            "mtp": n_mtp,
            "experts_bank": n_bank + n_alpha,
        },
        "copied_metadata": copied,
    })
    return index


__all__ = [
    "convert_checkpoint",
    "iter_converted_mtp_weights",
    "mtp_ftw_metadata",
]
