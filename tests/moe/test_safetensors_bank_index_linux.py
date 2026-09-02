"""GPU-free safetensors expert-bank index coverage.

These tests exercise Linux file mapping and positional-read behavior. The Python index
builder itself is portable, but production DISK serving and its UFFD option are Linux
features, so this suite is collected only on Linux.
"""

from __future__ import annotations

import json
import mmap
import os
import sys

import pytest

if sys.platform != "linux":
    pytest.skip("DISK bank serving is Linux-only", allow_module_level=True)

import torch
from safetensors.torch import save_file


def _write_checkpoint(path, layers):
    path.mkdir()
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": len(layers),
        "num_experts": layers[0][0].shape[0],
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weight_map = {}
    for layer_id, (gate_up, down) in enumerate(layers):
        shard = f"model-{layer_id + 1:05d}-of-{len(layers):05d}.safetensors"
        names = {
            f"model.layers.{layer_id}.mlp.experts.gate_up_proj": gate_up,
            f"model.layers.{layer_id}.mlp.experts.down_proj": down,
            f"model.layers.{layer_id}.input_layernorm.weight": torch.arange(
                7 + layer_id, dtype=torch.float32
            ),
        }
        save_file(names, str(path / shard))
        weight_map.update({name: shard for name in names})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


def _layers():
    return [
        (
            (torch.arange(3 * 5 * 7, dtype=torch.float32) + layer * 1000)
            .to(torch.bfloat16)
            .view(3, 5, 7),
            (torch.arange(3 * 7 * 2, dtype=torch.float32) + layer * 2000)
            .to(torch.bfloat16)
            .view(3, 7, 2),
        )
        for layer in range(2)
    ]


def test_index_build_roundtrip_records_each_expert_range(tmp_path):
    from freetoken.checkpoint.safetensors_bank_index import (
        INDEX_NAME,
        build_safetensors_bank_index,
    )

    checkpoint = tmp_path / "hf"
    layers = _layers()
    _write_checkpoint(checkpoint, layers)
    index = build_safetensors_bank_index(str(checkpoint))

    assert (checkpoint / INDEX_NAME).is_file()
    assert index["num_layers"] == 2
    assert index["num_experts"] == 3
    assert len(index["banks"]) == 4
    entry = next(
        item for item in index["banks"]
        if item["layer"] == 1 and item["bank"] == "gate_up"
    )
    assert len(entry["experts"]) == 3
    row_bytes = layers[1][0][0].numel() * layers[1][0].element_size()
    assert all(record[2] == row_bytes for record in entry["experts"])
    expected = layers[1][0][2].view(torch.uint8).numpy().tobytes()
    shard, offset, length = entry["experts"][2]
    with open(checkpoint / shard, "rb") as handle:
        handle.seek(offset)
        assert handle.read(length) == expected


def test_index_fingerprint_regenerates_after_shard_change(tmp_path):
    from freetoken.checkpoint.safetensors_bank_index import (
        ensure_safetensors_bank_index,
        is_safetensors_bank_index_stale,
    )

    checkpoint = tmp_path / "hf"
    _write_checkpoint(checkpoint, _layers())
    _folder, first = ensure_safetensors_bank_index(str(checkpoint))
    shard = checkpoint / first["shards"][0]["file"]
    stat = shard.stat()
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert is_safetensors_bank_index_stale(str(checkpoint), first)
    _folder, second = ensure_safetensors_bank_index(str(checkpoint))
    assert second["fingerprint"] != first["fingerprint"]
    assert not is_safetensors_bank_index_stale(str(checkpoint), second)


def test_misaligned_indexed_mapping_and_ftw_interface_parity(tmp_path):
    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name, load_ftw_banks
    from freetoken.checkpoint.safetensors_bank_index import (
        build_safetensors_bank_index,
        load_indexed_banks,
    )
    from freetoken.moe.host_banks import HostResidency

    checkpoint = tmp_path / "hf"
    layers = _layers()
    _write_checkpoint(checkpoint, layers)
    index = build_safetensors_bank_index(str(checkpoint))
    misaligned = next(item for item in index["banks"] if item["offset"] % mmap.PAGESIZE)

    indexed = load_indexed_banks(
        str(checkpoint),
        num_layers=2,
        dtype=torch.bfloat16,
        layer_residency=[HostResidency.DISK.value] * 2,
    )
    owner = indexed.sources[misaligned["bank"]][misaligned["layer"]]._freetoken_host_bank
    assert owner._map_offset == misaligned["offset"] // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    assert owner._view_offset == misaligned["offset"] - owner._map_offset
    row_bytes = misaligned["experts"][0][2]
    assert owner.populate_experts([2, 0, 2], bytearray(17)) == 2 * row_bytes

    ftw_dir = tmp_path / "ftw"
    writer = FTWWriter(str(ftw_dir))
    for layer_id, (gate_up, down) in enumerate(layers):
        writer.add_tensor(
            layer_bank_entry_name("gate_up", layer_id), gate_up, kind="experts_bank"
        )
        writer.add_tensor(
            layer_bank_entry_name("down", layer_id), down, kind="experts_bank"
        )
    writer.finalize({"quant_format": "bf16", "expert_bank_num_layers": 2})
    ftw = load_ftw_banks(
        str(ftw_dir),
        num_layers=2,
        layer_residency=[HostResidency.DISK.value] * 2,
    )

    for name in ("gate_up", "down"):
        for layer_id in range(2):
            assert torch.equal(indexed.sources[name][layer_id], ftw.sources[name][layer_id])
            assert (
                indexed.sources[name][layer_id].view(torch.uint8).numpy().tobytes()
                == ftw.sources[name][layer_id].view(torch.uint8).numpy().tobytes()
            )


def test_forced_index_rejects_repacked_architecture(tmp_path):
    from freetoken.checkpoint.safetensors_bank_index import (
        UnsupportedSafetensorsBankIndex,
        ensure_safetensors_bank_index,
    )

    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "num_hidden_layers": 1,
            "num_experts": 2,
        }),
        encoding="utf-8",
    )
    save_file({"model.weight": torch.ones(1)}, str(checkpoint / "model.safetensors"))
    with pytest.raises(UnsupportedSafetensorsBankIndex, match="byte-identical"):
        ensure_safetensors_bank_index(str(checkpoint))


def test_bank_source_cli_and_auto_detection(tmp_path):
    from freetoken.moe.expert_banks import resolve_bank_source
    from freetoken.server.args import parse_args

    checkpoint = tmp_path / "hf"
    _write_checkpoint(checkpoint, _layers())
    args, _ = parse_args([
        "--model", str(checkpoint),
        "--dtype", "bfloat16",
        "--bank-source", "index",
    ])
    assert args.bank_source == "index"
    assert resolve_bank_source(str(checkpoint), "auto") == "index"
