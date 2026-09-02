"""Resident and routed-expert loading for RedHatAI GLM-5.3-Flash-NVFP4.

The source routed layout is compressed-tensors NVFP4: ``weight_packed`` contains
two low-nibble-first E2M1 values per byte, ``weight_scale`` is one e4m3 scale per
group of 16 reduction values, and scalar ``weight_global_scale`` is the quant-side
global. The native FreeToken bank is byte-for-byte for packed/block-scale data and
stores ``1 / weight_global_scale`` expanded per output row. Triton serves that
layout directly; optional GPU backends repack it through the existing bank layer.
Activation ``input_global_scale`` tensors are W4A4 calibration data and are not
banked because FreeToken's expert kernels consume BF16 activations.
"""

from __future__ import annotations

import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm_moe import (
    glm_moe_bank_layer,
    validate_glm_nvfp4_bank_geometry,
)
from freetoken.models.loader import ShardReader, drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")
_NVFP4_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)


def _bank_layer(layer: int, config) -> int | None:
    if layer >= int(config.num_layers):
        return None
    mlp_types = tuple(getattr(config, "mlp_layer_types", ()) or ())
    if mlp_types and mlp_types[layer] != "sparse":
        return None
    return glm_moe_bank_layer(config, layer)


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_bank_layer,
    desc="GLM-5.3 compressed-tensors NVFP4 experts",
    kind_aliases={
        "weight_packed": "weight",
        "weight_scale": "weight_scale",
        "weight_global_scale": "weight_scale_2",
    },
    invert_global_scale=True,
)


def _rename(raw_name: str, num_layers: int = 45) -> str | None:
    """Checkpoint key to text-decoder key, or None for non-serving tensors."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.", "model.vision_")):
        return None
    match = _LAYER_RE.match(raw_name)
    if match is not None and int(match.group("layer")) >= num_layers:
        return None
    if _EXPERT_RE.search(raw_name):
        return None
    # The current DSA backend reuses glm_moe_dsa's token-level indexer. The newer
    # checkpoint's optional four-token k-pool compression is tracked as a hardware
    # validation item and its two resident tensors must not leak into strict loading.
    if raw_name.endswith(
        (
            ".indexer.index_kpool_compress_ape",
            ".indexer.index_kpool_compress_gate",
        )
    ):
        return None
    if raw_name.startswith("model.language_model."):
        name = "model." + raw_name[len("model.language_model.") :]
    elif raw_name.startswith("language_model."):
        name = "model." + raw_name[len("language_model.") :]
    else:
        name = raw_name
    return name.replace(
        ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if include_moe_experts:
        raise ValueError("glm5_next routed experts require an offload MoE backend")
    if not include_non_moe:
        return
    config = parse_config(cached_load_hf_config(model_path))
    reader = ShardReader(model_path, device)
    try:
        for file in tqdm(
            reader.files(),
            desc="Loading GLM-5.3 resident text weights",
            disable=not get_tp_info().is_primary(),
        ):
            for raw_name in reader.names_in(file):
                name = _rename(raw_name, config.num_layers)
                if name is not None:
                    yield name, reader.get_tensor(raw_name)
            drop_page_cache(file)
    finally:
        reader.close()


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    sources = load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )
    return validate_glm_nvfp4_bank_geometry(config, sources)


def load_nvfp4_expert_sources_parallel(
    model_path: str,
    config,
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
):
    sources = load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )
    return validate_glm_nvfp4_bank_geometry(config, sources)


__all__ = [
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
