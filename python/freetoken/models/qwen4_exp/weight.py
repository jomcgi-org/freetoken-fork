"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the FP8 or packed 4-bit n-gram table, either concatenated into
  pinned :class:`HostBank` objects or left as read-only shard mappings for disk staging or HMM.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped by the default reader: ``mtp.*`` (loaded separately when native MTP is enabled) and
``model.visual.*`` (served text-only).
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


def iter_mtp_weights(
    model_path: str,
    device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield only the optional BF16 MTP head under its native ``mtp.*`` keys."""
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp MTP weight loading supports TP=1 only")
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading MTP weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if not name.startswith("mtp."):
                    continue
                tensor = f.get_tensor(name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():
                        yield fused
                    continue
                yield name, tensor
    assert not fuse_buf, f"Incomplete MTP projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


_PLE_FORMATS = ("fp8", "int4g16", "e2m1g16")


@dataclass(frozen=True)
class PleTable:
    """A contiguous host PLE table and its format-specific scales."""

    bank: HostBank
    weight_scale: torch.Tensor
    format: str = "fp8"
    head_dim: int | None = None
    scale_bank: HostBank | None = None

    def __post_init__(self) -> None:
        if self.format not in _PLE_FORMATS:
            raise ValueError(f"unsupported PLE table format {self.format!r}")
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", int(self.bank.tensor.shape[1]))

    @property
    def tensor(self) -> torch.Tensor:
        """Raw table rows, packed to two elements per byte for 4-bit formats."""
        return self.bank.tensor

    @property
    def scales(self) -> torch.Tensor | None:
        return None if self.scale_bank is None else self.scale_bank.tensor

    @property
    def num_rows(self) -> int:
        return int(self.bank.tensor.shape[0])

    @property
    def row_nbytes(self) -> int:
        return self.bank.tensor.stride(0) * self.bank.tensor.element_size()


@dataclass(frozen=True)
class DiskPleTable:
    """PLE shards mapped for the staged disk and direct HMM backends."""

    banks: tuple[HostBank, ...]
    rows_per_shard: int
    weight_scale: torch.Tensor
    format: str = "fp8"
    logical_head_dim: int | None = None
    scale_banks: tuple[HostBank, ...] = ()

    def __post_init__(self) -> None:
        if self.format not in _PLE_FORMATS:
            raise ValueError(f"unsupported PLE table format {self.format!r}")
        if self.logical_head_dim is None:
            object.__setattr__(self, "logical_head_dim", int(self.banks[0].tensor.shape[1]))

    @property
    def num_rows(self) -> int:
        return len(self.banks) * self.rows_per_shard

    @property
    def head_dim(self) -> int:
        assert self.logical_head_dim is not None
        return self.logical_head_dim

    @property
    def row_nbytes(self) -> int:
        tensor = self.banks[0].tensor
        return tensor.stride(0) * tensor.element_size()


_PLE_ST_DTYPE = "F8_E4M3"
_PLE_PACKED_DTYPE = "U8"
_PLE_SHARD_SCALE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight_scale$"
)
_PLE_GLOBAL_SCALE_2_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale_2"


@dataclass(frozen=True)
class _PleTensorPart:
    path: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _PleTableLayout:
    parts: dict[int, _PleTensorPart]
    scale_parts: dict[int, _PleTensorPart]
    global_scales: torch.Tensor | None
    weight_scale: torch.Tensor
    rows: int
    cols: int
    stored_cols: int
    format: str
    data_dtype: torch.dtype
    stored_scale_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def _part(path: str, base: int, meta: dict) -> _PleTensorPart:
    begin, end = meta["data_offsets"]
    return _PleTensorPart(path, base + begin, end - begin)


def _sidecar_shard(path: str) -> int | None:
    match = re.fullmatch(r"shard_(\d+)\.safetensors", os.path.basename(path))
    return None if match is None else int(match.group(1))


def _ple_table_layout(model_path: str, qwen4_args) -> _PleTableLayout:
    """Resolve and validate PLE tensor payloads without reading their table bytes."""
    folder = download_hf_weight(model_path)
    data_meta: dict[int, tuple[str, dict, int]] = {}
    scale_meta: dict[int, tuple[str, dict, int]] = {}
    global_scale_meta: list[tuple[int | None, str, str, torch.Tensor]] = []
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        sidecar_shard = _sidecar_shard(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            match = _PLE_SHARD_RE.search(key)
            sidecar_data = key in ("weight_fp8", "weight_i4", "weight_e2m1")
            if match is not None or (sidecar_data and sidecar_shard is not None):
                shard = int(match.group("shard")) if match is not None else sidecar_shard
                if shard in data_meta:
                    raise ValueError(f"PLE table shard {shard} has multiple weight tensors")
                data_meta[shard] = (path, meta, base)
                continue
            match = _PLE_SHARD_SCALE_RE.search(key)
            sidecar_scale = key == "weight_scale" and sidecar_shard is not None
            if match is not None or sidecar_scale:
                shard = int(match.group("shard")) if match is not None else sidecar_shard
                if shard in scale_meta:
                    raise ValueError(f"PLE table shard {shard} has multiple scale tensors")
                scale_meta[shard] = (path, meta, base)
                continue
            if key.endswith((_PLE_SCALE_SUFFIX, _PLE_GLOBAL_SCALE_2_SUFFIX)) or (
                key == "weight_scale_2" and sidecar_shard is not None
            ):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    value = f.get_tensor(key)
                if value.numel() != 1:
                    raise ValueError(
                        f"PLE global scale {key} has shape {list(value.shape)}, expected scalar"
                    )
                global_scale_meta.append((sidecar_shard, path, key, value.reshape(())))

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(data_meta) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(data_meta)}: "
            f"{sorted(data_meta)[:8]}"
        )

    first_path, first_meta, _ = data_meta[0]
    first_shape = tuple(first_meta["shape"])
    if len(first_shape) != 2:
        raise ValueError(f"PLE table shard 0 has shape {list(first_shape)}, expected rank 2")
    rows, stored_cols = first_shape
    data_dtype = first_meta["dtype"]
    logical_cols = int(qwen4_args.ngram_head_dim)
    if data_dtype == _PLE_ST_DTYPE:
        table_format = "fp8"
        expected_stored_cols = logical_cols
        torch_data_dtype = torch.float8_e4m3fn
    elif data_dtype == _PLE_PACKED_DTYPE:
        table_format = ""
        expected_stored_cols = logical_cols // 2
        torch_data_dtype = torch.uint8
        if logical_cols % 16:
            raise ValueError(
                f"PLE group-16 table width {logical_cols} is not divisible by 16"
            )
    else:
        raise ValueError(
            f"PLE table shard 0 in {first_path} has unsupported dtype {data_dtype}; "
            f"expected {_PLE_ST_DTYPE} or {_PLE_PACKED_DTYPE}"
        )
    if stored_cols != expected_stored_cols:
        kind = "packed row" if data_dtype == _PLE_PACKED_DTYPE else "row"
        raise ValueError(
            f"PLE table {kind} is {stored_cols} wide, expected {expected_stored_cols} "
            f"for config width {logical_cols}"
        )
    for shard, (path, meta, _base) in data_meta.items():
        if meta["dtype"] != data_dtype:
            raise ValueError(
                f"PLE table shard {shard} in {path} has dtype {meta['dtype']}, "
                f"expected {data_dtype}"
            )
        if tuple(meta["shape"]) != (rows, stored_cols):
            raise ValueError(
                f"PLE table shard {shard} is {meta['shape']}, "
                f"expected {[rows, stored_cols]}"
            )

    stored_scale_dtype: torch.dtype | None = None
    scale_dtype: torch.dtype | None = None
    if scale_meta:
        if sorted(scale_meta) != list(range(expected)):
            raise ValueError(
                f"PLE table needs scale shards 0..{expected - 1}, found "
                f"{len(scale_meta)}: {sorted(scale_meta)[:8]}"
            )
        scale_st_dtype = scale_meta[0][1]["dtype"]
        if data_dtype == _PLE_ST_DTYPE:
            expected_scale_shape = (rows,)
            if scale_st_dtype != "F32":
                raise ValueError(
                    f"PLE fp8 row scales have dtype {scale_st_dtype}, expected F32"
                )
            stored_scale_dtype = scale_dtype = torch.float32
        else:
            expected_scale_shape = (rows, logical_cols // 16)
            if scale_st_dtype == "F16":
                table_format = "int4g16"
                stored_scale_dtype = scale_dtype = torch.float16
            elif scale_st_dtype == _PLE_ST_DTYPE:
                table_format = "e2m1g16"
                stored_scale_dtype = torch.float8_e4m3fn
                # weight_scale_2 is folded into these group scales below. FP8 cannot
                # represent the product accurately, so serving banks use FP16.
                scale_dtype = torch.float16
            else:
                raise ValueError(
                    f"PLE packed group scales have unsupported dtype {scale_st_dtype}; "
                    "expected F16 for INT4 or F8_E4M3 for e2m1"
                )
        for shard, (path, meta, _base) in scale_meta.items():
            if meta["dtype"] != scale_st_dtype:
                raise ValueError(
                    f"PLE scale shard {shard} in {path} has dtype {meta['dtype']}, "
                    f"expected {scale_st_dtype}"
                )
            if tuple(meta["shape"]) != expected_scale_shape:
                raise ValueError(
                    f"PLE scale shard {shard} is {meta['shape']}, "
                    f"expected {list(expected_scale_shape)}"
                )
    elif data_dtype == _PLE_PACKED_DTYPE:
        raise ValueError("PLE packed table has no per-shard group-16 weight_scale tensors")

    global_scales = None
    if table_format == "e2m1g16":
        by_shard: dict[int, torch.Tensor] = {}
        unassigned: list[tuple[str, str, torch.Tensor]] = []
        data_shards_by_path: dict[str, list[int]] = {}
        for shard, (path, _meta, _base) in data_meta.items():
            data_shards_by_path.setdefault(path, []).append(shard)
        for shard, path, key, value in global_scale_meta:
            if value.dtype != torch.float32:
                raise ValueError(
                    f"PLE e2m1 weight_scale_2 {key} in {path} has dtype "
                    f"{value.dtype}, expected float32"
                )
            if shard is None:
                candidates = data_shards_by_path.get(path, [])
                if len(candidates) == 1:
                    shard = candidates[0]
            if shard is None:
                unassigned.append((path, key, value))
            elif shard in by_shard:
                raise ValueError(f"PLE table shard {shard} has multiple weight_scale_2 tensors")
            else:
                by_shard[shard] = value
        if by_shard:
            if unassigned:
                path, key, _value = unassigned[0]
                raise ValueError(f"cannot associate PLE global scale {key} in {path} with a shard")
            if sorted(by_shard) != list(range(expected)):
                raise ValueError(
                    f"PLE table needs weight_scale_2 shards 0..{expected - 1}, found "
                    f"{len(by_shard)}: {sorted(by_shard)[:8]}"
                )
            global_scales = torch.stack([by_shard[shard] for shard in range(expected)])
        elif len(unassigned) == 1:
            # Backward compatibility for checkpoints with one table-wide packed scale.
            global_scales = unassigned[0][2].repeat(expected)
        elif unassigned and all(
            value.dtype == unassigned[0][2].dtype
            and torch.equal(value, unassigned[0][2])
            for _path, _key, value in unassigned[1:]
        ):
            # Older multi-file layouts did not expose a shard number for identical scales.
            global_scales = unassigned[0][2].repeat(expected)
        else:
            raise ValueError("PLE e2m1 group-16 table has no per-shard weight_scale_2")
        # The per-shard values are folded into FP16 group scales during bank loading.
        weight_scale = torch.tensor(1.0, dtype=torch.float32)
    elif global_scale_meta:
        weight_scale = global_scale_meta[0][3]
        for _shard, path, key, value in global_scale_meta[1:]:
            if value.dtype != weight_scale.dtype or not torch.equal(value, weight_scale):
                raise ValueError(f"PLE global scale {key} in {path} disagrees across shards")
    else:
        weight_scale = torch.tensor(1.0, dtype=torch.float32)
    if table_format == "fp8" and not scale_meta and not global_scale_meta:
        raise ValueError("PLE fp8 table has no weight_scale")
    if table_format == "int4g16" and global_scale_meta:
        raise ValueError("PLE INT4 group-16 table unexpectedly has a global weight_scale")
    if table_format == "fp8" and not scale_meta and weight_scale.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            f"PLE fp8 weight_scale has unsupported dtype {weight_scale.dtype}"
        )

    return _PleTableLayout(
        parts={
            shard: _part(path, base, meta)
            for shard, (path, meta, base) in data_meta.items()
        },
        scale_parts={
            shard: _part(path, base, meta)
            for shard, (path, meta, base) in scale_meta.items()
        },
        global_scales=global_scales,
        weight_scale=weight_scale,
        rows=rows,
        cols=logical_cols,
        stored_cols=stored_cols,
        format=table_format,
        data_dtype=torch_data_dtype,
        stored_scale_dtype=stored_scale_dtype,
        scale_dtype=scale_dtype,
    )


def load_ple_table(model_path: str, qwen4_args, *, backend: str = "pinned",
                   pin: bool = True, workers: int = 8,
                   chunk: int = 8 << 20) -> PleTable | DiskPleTable:
    """Load the PLE table as one pinned bank or read-only safetensors mappings.

    ``pinned`` preserves the original O_DIRECT concatenate, fill, then pin route. ``cached``,
    ``disk``, and ``hmm`` map each data payload from its safetensors file with ``MAP_SHARED``
    and ``MADV_RANDOM``. Folded E2M1 scales use anonymous FP16 banks. Their model backends
    differ in how rows reach the GPU.
    """
    if backend not in ("pinned", "cached", "disk", "hmm"):
        raise ValueError(
            f"--ple-backend must be 'pinned', 'cached', 'disk', or 'hmm', got {backend!r}"
        )
    layout = _ple_table_layout(model_path, qwen4_args)
    expected = int(qwen4_args.split_ngram_parts)
    shard_bytes = layout.rows * layout.stored_cols
    shard_bytes *= torch.empty((), dtype=layout.data_dtype).element_size()
    scale_shape = (
        (layout.rows,) if layout.format == "fp8" else (layout.rows, layout.cols // 16)
    )
    stored_scale_shard_bytes = 0
    if layout.scale_dtype is not None:
        assert layout.stored_scale_dtype is not None
        stored_scale_shard_bytes = math.prod(scale_shape) * torch.empty(
            (), dtype=layout.stored_scale_dtype
        ).element_size()
    scale_scratch = None
    if layout.global_scales is not None:
        assert layout.stored_scale_dtype is not None
        scale_scratch = HostBank(scale_shape, layout.stored_scale_dtype)

    def load_folded_scale_shard(shard: int, destination: torch.Tensor) -> None:
        assert scale_scratch is not None and layout.global_scales is not None
        scale_part = layout.scale_parts[shard]
        if scale_part.nbytes != stored_scale_shard_bytes:
            raise ValueError(
                f"PLE scale shard {shard} is {scale_part.nbytes} B, "
                f"expected {stored_scale_shard_bytes}"
            )
        read_range_into(
            scale_scratch.memoryview(), scale_part.path,
            file_offset=scale_part.offset, nbytes=scale_part.nbytes,
            workers=workers, chunk=chunk,
        )
        destination.copy_(scale_scratch.tensor)
        destination.mul_(float(layout.global_scales[shard]))
    if backend in ("cached", "disk", "hmm"):
        banks = []
        scale_banks = []
        for shard in range(expected):
            part = layout.parts[shard]
            if part.nbytes != shard_bytes:
                raise ValueError(
                    f"PLE shard {shard} is {part.nbytes} B, expected {shard_bytes}"
                )
            banks.append(
                HostBank(
                    (layout.rows, layout.stored_cols), layout.data_dtype, backing="file",
                    file_path=part.path, file_offset=part.offset,
                )
            )
            if layout.scale_dtype is not None:
                if layout.global_scales is None:
                    scale_part = layout.scale_parts[shard]
                    if scale_part.nbytes != stored_scale_shard_bytes:
                        raise ValueError(
                            f"PLE scale shard {shard} is {scale_part.nbytes} B, "
                            f"expected {stored_scale_shard_bytes}"
                        )
                    scale_banks.append(
                        HostBank(
                            scale_shape, layout.scale_dtype, backing="file",
                            file_path=scale_part.path, file_offset=scale_part.offset,
                        )
                    )
                else:
                    scale_bank = HostBank(scale_shape, layout.scale_dtype)
                    load_folded_scale_shard(shard, scale_bank.tensor)
                    scale_banks.append(scale_bank)
        return DiskPleTable(
            tuple(banks), layout.rows, layout.weight_scale, layout.format,
            layout.cols, tuple(scale_banks),
        )

    bank = HostBank((expected * layout.rows, layout.stored_cols), layout.data_dtype)
    scale_bank = None
    if layout.scale_dtype is not None:
        scale_bank = HostBank(
            (expected * layout.rows, *scale_shape[1:]), layout.scale_dtype
        )
    bar = byte_bar(
        expected * (shard_bytes + stored_scale_shard_bytes), "Loading PLE table"
    )
    try:
        buf = bank.memoryview()
        scale_buf = None if scale_bank is None else scale_bank.memoryview()
        for shard in range(expected):
            part = layout.parts[shard]
            assert part.nbytes == shard_bytes, (
                f"PLE shard {shard} is {part.nbytes} B, expected {shard_bytes}"
            )
            read_range_into(buf, part.path, file_offset=part.offset, nbytes=part.nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            if scale_buf is not None:
                scale_part = layout.scale_parts[shard]
                if layout.global_scales is None:
                    read_range_into(
                        scale_buf, scale_part.path, file_offset=scale_part.offset,
                        nbytes=scale_part.nbytes, dest_offset=shard * scale_part.nbytes,
                        workers=workers, chunk=chunk,
                    )
                else:
                    assert scale_bank is not None
                    rows = scale_bank.tensor[
                        shard * layout.rows : (shard + 1) * layout.rows
                    ]
                    load_folded_scale_shard(shard, rows)
            bar.update(
                part.nbytes + (0 if scale_buf is None else stored_scale_shard_bytes)
            )
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
        if scale_bank is not None:
            scale_bank.pin()
    return PleTable(
        bank=bank,
        weight_scale=layout.weight_scale,
        format=layout.format,
        head_dim=layout.cols,
        scale_bank=scale_bank,
    )


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "PleTable",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
