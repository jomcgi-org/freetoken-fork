from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from freetoken.utils import download_hf_weight
from tqdm import tqdm


# Native ModelOpt NVFP4 values. The serving kernels use the same nibble ordering and
# dequantization rule in kernel/triton/nvfp4_{dequant,fused_moe}.py.
_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str


def quantize_nvfp4_group16(
    weight: torch.Tensor,
    *,
    row_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize dense rows to the native group-16 NVFP4 expert layout.

    The last dimension is the reduction dimension. Two E2M1 values are packed in
    low-nibble-first order, every 16 values share an e4m3 scale, and every output row
    carries one fp16 global scale. This is the same six-bank representation used by the
    main routed experts, so the resident MTP head can use the existing inline-dequant MoE
    kernels without a BF16 staging copy.

    Conversion is chunked by flattened output row. A full Qwen3.8 MTP expert tensor is
    several GiB, so even the float32 conversion workspace must stay bounded.
    """
    if weight.ndim < 2:
        raise ValueError(f"NVFP4 weight must have at least 2 dimensions, got {weight.shape}")
    in_features = int(weight.shape[-1])
    if in_features % 16:
        raise ValueError(
            f"NVFP4 group-16 reduction dimension must be divisible by 16, got {in_features}"
        )
    rows = weight.detach().to(device="cpu").reshape(-1, in_features)
    packed = torch.empty((rows.shape[0], in_features // 2), dtype=torch.uint8)
    scales = torch.empty(
        (rows.shape[0], in_features // 16), dtype=torch.float8_e4m3fn
    )
    globals_ = torch.empty((rows.shape[0],), dtype=torch.float16)
    # Midpoints between the eight non-negative E2M1 magnitudes. bucketize maps a
    # normalized magnitude directly to its nearest code without a 16x codebook-distance
    # tensor. Exact midpoint ties choose the lower code deterministically.
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

    for start in range(0, rows.shape[0], row_chunk):
        end = min(start + row_chunk, rows.shape[0])
        groups = rows[start:end].float().reshape(end - start, -1, 16)
        desired_scale = groups.abs().amax(dim=-1) / _E2M1_MAX
        # Use the full e4m3 range for the largest group in each row. Zero rows use a
        # benign global of one and zero block scales/codes.
        row_global = desired_scale.amax(dim=-1) / _E4M3_MAX
        row_global = torch.where(row_global == 0, torch.ones_like(row_global), row_global)
        row_global16 = row_global.to(torch.float16)
        if torch.any(row_global16 == 0):
            raise ValueError("NVFP4 row global scale underflowed float16 during conversion")
        normalized_scale = desired_scale / row_global16.float().unsqueeze(-1)
        scale8 = normalized_scale.clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        actual_scale = scale8.float() * row_global16.float().unsqueeze(-1)
        safe_scale = torch.where(actual_scale == 0, torch.ones_like(actual_scale), actual_scale)
        normalized = groups / safe_scale.unsqueeze(-1)
        magnitude_codes = torch.bucketize(normalized.abs().contiguous(), thresholds)
        codes = (magnitude_codes + (normalized < 0) * 8).to(torch.uint8)
        codes = torch.where(actual_scale.unsqueeze(-1) == 0, torch.zeros_like(codes), codes)
        codes = codes.reshape(end - start, in_features)
        packed[start:end] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[start:end] = scale8
        globals_[start:end] = row_global16

    leading = tuple(weight.shape[:-1])
    return (
        packed.reshape(*leading, in_features // 2),
        scales.reshape(*leading, in_features // 16),
        globals_.reshape(*leading),
    )


def dequantize_nvfp4_group16(
    packed: torch.Tensor,
    scale: torch.Tensor,
    row_global: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """CPU/reference dequantization for :func:`quantize_nvfp4_group16`."""
    if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected uint8/e4m3 NVFP4 tensors, got {packed.dtype}/{scale.dtype}")
    in_features = int(packed.shape[-1]) * 2
    if scale.shape != (*packed.shape[:-1], in_features // 16):
        raise ValueError(f"NVFP4 scale shape {scale.shape} does not match packed {packed.shape}")
    if row_global.shape != packed.shape[:-1]:
        raise ValueError(
            f"NVFP4 global shape {row_global.shape} does not match packed {packed.shape}"
        )
    codes = torch.empty((*packed.shape[:-1], in_features), dtype=torch.long)
    codes[..., 0::2] = packed & 0xF
    codes[..., 1::2] = packed >> 4
    values = _E2M1_VALUES[codes]
    block_scale = scale.float().repeat_interleave(16, dim=-1)
    return (values * block_scale * row_global.float().unsqueeze(-1)).to(dtype)


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.tensor`` then pins it (per-layer, via
    ``PinPipeline``, as its writes complete)."""
    from freetoken.moe.host_banks import alloc_layer_banks

    fp8 = torch.float8_e4m3fn
    return alloc_layer_banks({
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), fp8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), fp8),
        "down_global": ((E, H), torch.float16),
    }, num_layers)


def load_nvfp4_expert_source_banks(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Build the 6 native NVFP4 source banks by streaming checkpoint shards (serial per-shard read).

    ModelOpt row layout: gate/up fused on the output-row axis, down separate; the per-tensor
    global scale (weight_scale_2) is kept as a separate per-output-row FP16 bank (``*_global``),
    so dequant is ``fp4 * block_scale * global``. Each bank is one ``[E, ...]`` tensor per
    layer, indexed by ``[bank_layer][expert]``. (The marlin/b12x backends repack these and
    fold the global into per-expert alphas; see moe/nvfp4_backends.py.)

    ``layer_sink=None`` (serving): pin each bank layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter; for
    marlin/b12x the provider wraps it in a per-layer repacking sink first): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    folder = download_hf_weight(model_path)
    index_path = os.path.join(folder, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    weight_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    global_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = _bank_layer(spec, layer, config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = match.group("kind")
        if kind == "weight_scale_2":
            global_shards[shard].append((name, match, bank_layer))
        elif kind in {"weight", "weight_scale"}:
            weight_shards[shard].append((name, match, bank_layer))
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_shards):
        path = os.path.join(folder, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, match, _bank_layer_id in global_shards[shard]:
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("proj"),
                )
                globals_map[key] = f.get_tensor(name).to(torch.float16)
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for shard in tqdm(sorted(weight_shards), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match, bank_layer_id in weight_shards[shard]:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    kind = match.group("kind")
                    tensor = f.get_tensor(name)
                    if kind == "weight":
                        if role == "gate":
                            gate_up_packed[bank_layer_id][expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[bank_layer_id][expert, I:] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id][expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id][expert, :I] = tensor
                            gate_up_global[bank_layer_id][expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[bank_layer_id][expert, I:] = tensor
                            gate_up_global[bank_layer_id][expert, I:] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id][expert] = tensor
                            down_global[bank_layer_id][expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    tracker.note(bank_layer_id)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def load_nvfp4_expert_source_banks_parallel(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel counterpart of :func:`load_nvfp4_expert_source_banks`, byte-for-byte same
    placement. bulk weight/weight_scale read via chunked multi-threaded O_DIRECT reader
    (iter_expert_tensors_parallel); tiny globals (``weight_scale_2``) stay serial (negligible
    bytes). ``layer_sink``: see :func:`load_nvfp4_expert_source_banks`."""
    from freetoken.models.weight import iter_expert_tensors_parallel

    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    weight_info: dict[str, tuple[re.Match[str], int]] = {}  # name -> (match, bank_layer)
    global_names_by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        kind = match.group("kind")
        if kind == "weight_scale_2":
            global_names_by_shard[shard].append(name)
        elif kind in {"weight", "weight_scale"}:
            weight_info[name] = (match, bank_layer)
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    # Pass 1: tiny per-tensor global scales (serial; data is scalar-per-expert).
    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_names_by_shard):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in global_names_by_shard[shard]:
                m = spec.key_pattern.match(name)
                globals_map[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = (
                    f.get_tensor(name).to(torch.float16)
                )
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    # Pass 2: bulk weight/weight_scale via the common parallel reader; place by name.
    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for name, tensor in iter_expert_tensors_parallel(
            folder, lambda n: n in weight_info, workers=workers, chunk=chunk
        ):
            match, bank_layer_id = weight_info[name]
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role[proj]
            kind = match.group("kind")
            if kind == "weight":
                if role == "gate":
                    gate_up_packed[bank_layer_id][expert, :I] = tensor
                elif role == "up":
                    gate_up_packed[bank_layer_id][expert, I:] = tensor
                else:
                    down_packed[bank_layer_id][expert] = tensor
            else:
                g = globals_map[(layer, expert, proj)]
                if role == "gate":
                    gate_up_scale[bank_layer_id][expert, :I] = tensor
                    gate_up_global[bank_layer_id][expert, :I] = g
                elif role == "up":
                    gate_up_scale[bank_layer_id][expert, I:] = tensor
                    gate_up_global[bank_layer_id][expert, I:] = g
                else:
                    down_scale[bank_layer_id][expert] = tensor
                    down_global[bank_layer_id][expert] = g
            tracker.note(bank_layer_id)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


__all__ = [
    "Nvfp4ExpertSourceSpec",
    "dequantize_nvfp4_group16",
    "load_nvfp4_expert_source_banks",
    "load_nvfp4_expert_source_banks_parallel",
    "quantize_nvfp4_group16",
]
