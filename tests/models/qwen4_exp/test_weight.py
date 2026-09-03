"""qwen4_exp weight loading against a synthetic checkpoint shaped like the RadixArk NVFP4 one.

The tensors are tiny but the key names, dtypes and the fusion geometry that matters
(hc_lowrank=320 + hc_count=4 -> a 12-row zero pad) are the real ones.
"""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes
from freetoken.models.qwen4_exp.weight import (
    _ZERO_CENTERED_NORM_SUFFIXES,
    _ple_table_layout,
    iter_weights,
    iter_mtp_weights,
    load_ple_table,
)
from freetoken.moe.host_banks import HostBank, read_range_into

from .common import requires_cuda

H = 32  # hidden_size
HC = 4  # hc_count
LR = 320  # hc_lowrank; kept real so the merged HC pad is the real (-(320+4)) % 16 = 12
HCH = HC * H  # hyper-connection stream width
KH, VH, HD = 2, 6, 8  # GDN key / value heads, head dim
QH, KVH, AHD = 4, 2, 16  # QSA q / kv heads, head dim
IHD = 8  # indexer head dim
E, I = 3, 6  # routed experts, moe_intermediate_size
NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4
QUANT_NGRAM_DIM = 32


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    """Layer 0 = GDN + PLE, layer 1 = QSA; plus the mtp / visual / routed-expert noise."""
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, I),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
        for expert in range(E):
            base = f"{lm}.layers.{layer}.mlp.experts.{expert}"
            for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                raw[f"{base}.{proj}.weight"] = torch.randint(
                    0, 256, (out, inn // 2), dtype=torch.uint8
                )
                raw[f"{base}.{proj}.weight_scale"] = torch.ones(
                    out, inn // 16 or 1, dtype=torch.float8_e4m3fn
                )
                raw[f"{base}.{proj}.weight_scale_2"] = torch.tensor(0.5)
                raw[f"{base}.{proj}.input_scale"] = torch.tensor(0.25)
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, H),
        f"{ple}.value_proj.weight": _bf16(H, H),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    raw.update(_hc_weights("mtp.hyper_connection_mixer", inject=False))
    raw.update(_hc_weights("mtp.layers.0.attn_hyper_connection", inject=True))
    raw.update(_hc_weights("mtp.layers.0.mlp_hyper_connection", inject=True))
    raw.update({
        "mtp.fc_embedding.weight": _bf16(H, H),
        "mtp.fc_hidden.weight": _bf16(H, H),
        "mtp.pre_fc_norm_embedding.weight": _bf16(H),
        "mtp.pre_fc_norm_hidden.weight": _bf16(HCH),
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "mtp.layers.0.self_attn.k_proj.weight": _bf16(KVH * AHD, H),
        "mtp.layers.0.self_attn.v_proj.weight": _bf16(KVH * AHD, H),
        "mtp.layers.0.self_attn.o_proj.weight": _bf16(H, QH * AHD),
        "mtp.layers.0.self_attn.q_norm.weight": _bf16(AHD),
        "mtp.layers.0.self_attn.k_norm.weight": _bf16(AHD),
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": _bf16(IHD),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": _bf16(IHD),
        "mtp.layers.0.mlp.gate.weight": _bf16(E, H),
        "mtp.layers.0.mlp.experts.gate_up_proj": _bf16(E, 2 * I, H),
        "mtp.layers.0.mlp.experts.down_proj": _bf16(E, H, I),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": _bf16(I, H),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": _bf16(H, I),
        "mtp.layers.0.mlp.shared_expert_gate.weight": _bf16(1, H),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
        "model.visual.merger.norm.weight": _bf16(H),
    })
    return raw


def _ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM).view(torch.float8_e4m3fn)
        )
        for i in range(NGRAM_SHARDS)
    }
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    shards[f"{prefix}.weight_scale"] = scale
    return shards, scale


def _quantized_ple_checkpoint(
    folder, table_format: str, *, identical_global_scales: bool = False
):
    """Write published-layout sidecar shards and return a torch dequant reference."""
    generator = torch.Generator().manual_seed(31)
    references = []
    for shard in range(NGRAM_SHARDS):
        if table_format == "fp8":
            data = (
                torch.randn(NGRAM_ROWS, QUANT_NGRAM_DIM, generator=generator) * 0.5
            ).to(torch.float8_e4m3fn)
            scales = torch.rand(NGRAM_ROWS, generator=generator, dtype=torch.float32) + 0.25
            tensors = {"weight_fp8": data, "weight_scale": scales}
            reference = data.float() * scales[:, None]
        else:
            codes = torch.randint(
                0, 16, (NGRAM_ROWS, QUANT_NGRAM_DIM), generator=generator, dtype=torch.uint8
            )
            data = codes[:, 0::2] | (codes[:, 1::2] << 4)
            if table_format == "int4g16":
                scales = torch.rand(
                    NGRAM_ROWS, QUANT_NGRAM_DIM // 16, generator=generator,
                    dtype=torch.float16,
                )
                tensors = {"weight_i4": data, "weight_scale": scales}
                reference = (codes.float() - 8) * scales.float().repeat_interleave(16, 1)
            else:
                scales = (
                    torch.rand(
                        NGRAM_ROWS, QUANT_NGRAM_DIM // 16, generator=generator
                    ) * 0.5 + 0.25
                ).to(torch.float8_e4m3fn)
                global_scale = torch.tensor(
                    0.75 if identical_global_scales else 0.5 + shard * 0.25,
                    dtype=torch.float32,
                )
                tensors = {
                    "weight_e2m1": data,
                    "weight_scale": scales,
                    "weight_scale_2": global_scale,
                }
                lut = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
                )
                reference = lut[codes.long()] * scales.float().repeat_interleave(16, 1)
                reference *= global_scale
        save_file(tensors, str(folder / f"shard_{shard}.safetensors"))
        references.append(reference.to(torch.bfloat16))
    return torch.cat(references)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    folder = tmp_path_factory.mktemp("qwen4_exp_ckpt")
    raw = _raw_checkpoint()
    table, _scale = _ngram_table()
    # Spread the dense tensors over two shards so the fusion buffer has to survive a file
    # boundary, and put the n-gram table in its own shards like the real checkpoint does.
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    return str(folder), {**raw, **table}


@pytest.fixture(scope="module")
def loaded(checkpoint) -> dict[str, torch.Tensor]:
    folder, _raw = checkpoint
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }


@pytest.fixture(scope="module")
def loaded_mtp(checkpoint) -> dict[str, torch.Tensor]:
    folder, _raw = checkpoint
    return dict(iter_mtp_weights(folder, torch.device("cpu")))


def _expected_names() -> set[str]:
    names = {"model.embed_tokens.weight", "lm_head.weight"}
    names |= {f"model.hyper_connection_mixer.{leaf}" for leaf in
              ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")}
    for layer in (0, 1):
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            names |= {f"model.layers.{layer}.{hc}.{leaf}" for leaf in (
                "hc_norm.weight", "input_mix_weight_down_block_inject.weight",
                "input_mix_weight_up.weight")}
        names |= {f"model.layers.{layer}.mlp.{leaf}" for leaf in (
            "gate.weight", "shared_expert.gate_up_proj.weight",
            "shared_expert.down_proj.weight", "shared_expert_gate.weight")}
    names |= {f"model.layers.0.linear_attn.{leaf}" for leaf in (
        "in_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")}
    names |= {f"model.layers.0.ple.{leaf}" for leaf in (
        "key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight",
        "norm_conv.weight", "conv1d.weight", "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes")}
    names |= {f"model.layers.1.self_attn.{leaf}" for leaf in (
        "qkv_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
        "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight",
        "indexer.k_layernorm.weight")}
    return names


def test_key_map_is_exactly_the_model_state_dict(loaded):
    assert set(loaded) == _expected_names()


def test_mtp_visual_experts_and_table_never_loaded(loaded):
    for name in loaded:
        assert not name.startswith(("mtp.", "model.visual."))
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))


def test_mtp_reader_loads_head_only_and_fuses_checkpoint_parts(loaded_mtp, checkpoint):
    _folder, raw = checkpoint
    assert loaded_mtp
    assert all(name.startswith("mtp.") for name in loaded_mtp)
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in loaded_mtp
    assert torch.equal(
        loaded_mtp["mtp.layers.0.mlp.experts.gate_up_proj"],
        raw["mtp.layers.0.mlp.experts.gate_up_proj"],
    )
    qkv = loaded_mtp["mtp.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.shape == (2 * QH * AHD + 2 * KVH * AHD, H)
    hc = loaded_mtp[
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    ]
    assert hc.shape == (LR + HC + 12, HCH)
    assert torch.count_nonzero(hc[-12:]) == 0


def test_hc_merge_is_down_then_inject_then_zero_pad(loaded, checkpoint):
    _folder, raw = checkpoint
    key = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    merged = loaded[key]
    assert merged.shape == (LR + HC + 12, HCH)  # pad = (-(320 + 4)) % 16
    down = raw["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"]
    inject = raw["model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(merged[:LR], down)
    assert torch.equal(merged[LR:LR + HC], inject)
    assert torch.equal(merged[LR + HC:], torch.zeros(12, HCH, dtype=merged.dtype))


def test_top_level_mixer_keeps_the_unmerged_down(loaded, checkpoint):
    _folder, raw = checkpoint
    got = loaded["model.hyper_connection_mixer.input_mix_weight_down.weight"]
    assert got.shape == (LR, HCH)
    assert torch.equal(
        got, raw["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"]
    )
    assert torch.equal(
        loaded["model.hyper_connection_mixer.input_mix_weight_up.weight"],
        raw["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"],
    )


def test_qkv_fusion_slices_back_to_q_k_v(loaded, checkpoint):
    _folder, raw = checkpoint
    attn = "model.language_model.layers.1.self_attn"
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in ("q", "k", "v")]
    fused = loaded["model.layers.1.self_attn.qkv_proj.weight"]
    assert fused.shape == (2 * QH * AHD + 2 * KVH * AHD, H)  # q carries the output gate
    for part, back in zip(parts, torch.split(fused, [p.shape[0] for p in parts], dim=0)):
        assert torch.equal(part, back)


def test_gdn_in_proj_slices_round_trip(loaded, checkpoint):
    _folder, raw = checkpoint
    gdn = "model.language_model.layers.0.linear_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    fused = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert fused.shape == (sum(p.shape[0] for p in parts), H)
    splits = torch.split(fused, [p.shape[0] for p in parts], dim=0)
    for part, back in zip(parts, splits):
        assert torch.equal(part, back)


def test_shared_expert_gate_up_merge(loaded, checkpoint):
    _folder, raw = checkpoint
    base = "model.language_model.layers.1.mlp.shared_expert"
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(merged[:I], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(merged[I:], raw[f"{base}.up_proj.weight"])


ZERO_CENTERED = (
    "model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.layers.0.mlp_hyper_connection.hc_norm.weight",
    "model.hyper_connection_mixer.hc_norm.weight",
    "model.layers.0.ple.norm_key.weight",
    "model.layers.0.ple.norm_query.weight",
    "model.layers.0.ple.norm_conv.weight",
    "model.layers.1.self_attn.q_norm.weight",
    "model.layers.1.self_attn.k_norm.weight",
    "model.layers.1.self_attn.indexer.q_layernorm.weight",
    "model.layers.1.self_attn.indexer.k_layernorm.weight",
)


def test_zero_centered_norms_are_loaded_raw(loaded, checkpoint):
    """(1+w) is applied at runtime in fp32, so the loader must not fold it into the bf16 weight."""
    _folder, raw = checkpoint
    for name in ZERO_CENTERED:
        raw_name = name.replace("model.", "model.language_model.", 1)
        assert torch.equal(loaded[name], raw[raw_name]), name


def test_the_zero_centered_suffix_list_covers_every_such_norm():
    assert {n for n in ZERO_CENTERED if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)} == set(ZERO_CENTERED)
    assert not "model.layers.0.linear_attn.norm.weight".endswith(_ZERO_CENTERED_NORM_SUFFIXES)


def test_gdn_gated_norm_passes_through(loaded, checkpoint):
    _folder, raw = checkpoint
    assert torch.equal(
        loaded["model.layers.0.linear_attn.norm.weight"],
        raw["model.language_model.layers.0.linear_attn.norm.weight"],
    )


def test_hash_constants_stay_int64(loaded):
    for leaf in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        assert loaded[f"model.layers.0.ple.ple_embedding.{leaf}"].dtype is torch.int64


def test_load_ple_table_concatenates_shards_in_index_order(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_ple_table(folder, args, pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{prefix}.shard_{shard}.weight"].view(torch.uint8))
    assert table.weight_scale.dtype is torch.bfloat16
    assert float(table.weight_scale) == 0.125


@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_quantized_ple_layout_detection_and_cpu_reference(tmp_path, table_format):
    from freetoken.models.qwen4_exp.ple import dequantize_ple_rows

    reference = _quantized_ple_checkpoint(tmp_path, table_format)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    table = load_ple_table(str(tmp_path), args, pin=False)

    assert table.format == table_format
    assert table.num_rows == NGRAM_SHARDS * NGRAM_ROWS
    assert table.head_dim == QUANT_NGRAM_DIM
    expected_row_nbytes = QUANT_NGRAM_DIM if table_format == "fp8" else QUANT_NGRAM_DIM // 2
    assert table.row_nbytes == expected_row_nbytes
    assert table.scales is not None
    if table_format == "e2m1g16":
        layout = _ple_table_layout(str(tmp_path), args)
        assert torch.equal(
            layout.global_scales,
            torch.tensor([0.5, 0.75, 1.0, 1.25], dtype=torch.float32),
        )
        assert table.scales.dtype is torch.float16
        assert float(table.weight_scale) == 1.0
    got = dequantize_ple_rows(
        table.tensor, table.scales, table.format, float(table.weight_scale)
    )
    assert torch.equal(got, reference)


def test_e2m1_ple_identical_global_scales_are_folded(tmp_path):
    from freetoken.models.qwen4_exp.ple import dequantize_ple_rows

    reference = _quantized_ple_checkpoint(
        tmp_path, "e2m1g16", identical_global_scales=True
    )
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    table = load_ple_table(str(tmp_path), args, pin=False)

    assert table.scales is not None and table.scales.dtype is torch.float16
    assert float(table.weight_scale) == 1.0
    got = dequantize_ple_rows(
        table.tensor, table.scales, table.format, float(table.weight_scale)
    )
    assert torch.equal(got, reference)


def test_ple_scope_names_final_banks_and_marks_scale_scratch_transient(
    tmp_path, monkeypatch,
):
    import freetoken.models.qwen4_exp.weight as weight

    from freetoken.moe.host_banks import requested_hugepages

    _quantized_ple_checkpoint(tmp_path, "e2m1g16")
    monkeypatch.setattr(weight, "read_range_into", lambda *_args, **_kwargs: 0)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    with requested_hugepages("off") as scope:
        table = load_ple_table(str(tmp_path), args, pin=False)

    reported = {
        tensor._freetoken_host_bank for tensor in scope.sources["PLE table"]
    }
    assert reported == {table.bank, table.scale_bank}
    transient = {
        bank for bank in scope.banks.values()
        if bank._hugepage_status["transient"]
    }
    assert len(transient) == 1
    assert reported.isdisjoint(transient)


@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_quantized_ple_staging_uses_packed_stride_and_matches_reference(
    tmp_path, table_format
):
    from freetoken.models.qwen4_exp.ple import DiskStagedTable

    reference = _quantized_ple_checkpoint(tmp_path, table_format)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    table = load_ple_table(str(tmp_path), args, backend="disk")
    backend = DiskStagedTable(
        table,
        stage_capacity_rows=8,
        device=torch.device("cpu"),
        prefetch=False,
        max_decode_batch_size=2,
        rows_per_token=3,
    )
    ids = torch.tensor([[0, 9, 0], [table.num_rows - 1, 9, 3]])

    assert backend._row_nbytes == table.row_nbytes
    if table_format != "fp8":
        assert backend._row_nbytes == QUANT_NGRAM_DIM // 2
    backend.prepare_decode(ids)
    got = backend.lookup(torch.zeros_like(ids))
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)
    assert torch.equal(got, want)
    backend.finish_decode(record_event=False)


def test_disk_ple_mapping_matches_pinned_rows(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    pinned = load_ple_table(folder, args, pin=False)
    disk = load_ple_table(folder, args, backend="disk")

    assert len(disk.banks) == NGRAM_SHARDS
    assert disk.rows_per_shard == NGRAM_ROWS
    assert any(bank._view_offset for bank in disk.banks)
    for row in (0, 1, NGRAM_ROWS - 1, NGRAM_ROWS, disk.num_rows - 1):
        shard, local = divmod(row, NGRAM_ROWS)
        assert torch.equal(
            disk.banks[shard].tensor[local].view(torch.uint8),
            pinned.tensor[row].view(torch.uint8),
        )


def test_hmm_ple_loader_reuses_unpinned_disk_mappings(checkpoint):
    from freetoken.models.qwen4_exp.ple import hmm_row_address

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    hmm = load_ple_table(folder, args, backend="hmm")

    assert len(hmm.banks) == NGRAM_SHARDS
    assert hmm.rows_per_shard == NGRAM_ROWS
    assert all(bank._disk and not bank._pinned for bank in hmm.banks)
    bases = tuple(bank.tensor.data_ptr() for bank in hmm.banks)
    assert hmm_row_address(bases, NGRAM_ROWS, NGRAM_ROWS, NGRAM_DIM) == bases[1]


def test_hmm_shard_base_addressing_crosses_boundaries():
    from freetoken.models.qwen4_exp.ple import hmm_row_address

    bases = (0x100000, 0x240000, 0x390000, 0x520000)
    rows_per_shard = 7
    row_nbytes = 160
    assert hmm_row_address(bases, 0, rows_per_shard, row_nbytes) == bases[0]
    assert hmm_row_address(bases, 6, rows_per_shard, row_nbytes) == bases[0] + 6 * 160
    assert hmm_row_address(bases, 7, rows_per_shard, row_nbytes) == bases[1]
    assert hmm_row_address(bases, 27, rows_per_shard, row_nbytes) == bases[3] + 6 * 160
    with pytest.raises(IndexError):
        hmm_row_address(bases, 28, rows_per_shard, row_nbytes)


def test_disk_ple_staging_matches_direct_rows(checkpoint):
    from freetoken.models.qwen4_exp.ple import DiskStagedTable

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    pinned = load_ple_table(folder, args, pin=False)
    disk = load_ple_table(folder, args, backend="disk")
    backend = DiskStagedTable(
        disk,
        stage_capacity_rows=8,
        device=torch.device("cpu"),
        prefetch=False,
        max_decode_batch_size=2,
        rows_per_token=3,
    )
    ids = torch.tensor(
        [[0, NGRAM_ROWS + 2, 0], [disk.num_rows - 1, NGRAM_ROWS + 2, 3]],
        dtype=torch.int64,
    )
    backend.prepare_decode(ids)
    inverse = backend.local_ids[: ids.shape[0]].long()
    staged = backend._stage_bank.tensor[: torch.unique(ids).numel()]

    got = staged.view(torch.uint8).index_select(0, inverse.reshape(-1)).view(
        *ids.shape, NGRAM_DIM
    )
    want = pinned.tensor.view(torch.uint8).index_select(0, ids.reshape(-1)).view_as(got)
    assert torch.equal(got, want)
    want_lookup = want.view(torch.float8_e4m3fn).to(torch.bfloat16) * backend.scale
    want_lookup = want_lookup.view(ids.shape[0], -1)
    assert torch.equal(backend.lookup(torch.zeros_like(ids)), want_lookup)
    backend.finish_decode(record_event=False)


def test_disk_ple_decode_fast_path_microbenchmark():
    """One thousand full host hash/stage steps stay bit-exact with the scalar oracle."""
    from freetoken.models.qwen4_exp.ple import (
        DiskStagedTable,
        NGramEmbedding,
        _derive_decode_row_ids_host_reference,
    )

    steps = 1000
    rows_per_shard = 64
    shard_count = 128
    head_dim = 160
    heads_per_ngram = 8
    num_heads = 16
    eos_token_id = 2
    banks = tuple(
        HostBank((rows_per_shard, head_dim), torch.float8_e4m3fn)
        for _ in range(shard_count)
    )
    for shard, bank in enumerate(banks):
        values = torch.arange(bank.tensor.numel(), dtype=torch.int64)
        bank.tensor.view(torch.uint8).view(-1).copy_(
            ((values + shard * 37) % 251).to(torch.uint8)
        )
    table = SimpleNamespace(
        banks=banks,
        rows_per_shard=rows_per_shard,
        num_rows=rows_per_shard * shard_count,
        head_dim=head_dim,
        weight_scale=1.0,
    )
    reference_backend = DiskStagedTable(
        table,
        stage_capacity_rows=num_heads,
        device=torch.device("cpu"),
        prefetch=False,
        max_decode_batch_size=1,
        rows_per_token=num_heads,
    )
    fast_backend = DiskStagedTable(
        table,
        stage_capacity_rows=num_heads,
        device=torch.device("cpu"),
        prefetch=False,
        max_decode_batch_size=1,
        rows_per_token=num_heads,
    )
    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=heads_per_ngram,
        num_ngram_heads=num_heads,
        ngram_boundary_token_id=eos_token_id,
    )
    embedding = NGramEmbedding(args)
    multipliers = torch.tensor(
        [6_364_136_223_846_793_005, 1_442_695_040_888_963_407, 3_202_034_522_624_059_733],
        dtype=torch.int64,
    )
    sizes = torch.tensor(
        [487, 491, 499, 503, 509, 479, 467, 463] * 2, dtype=torch.int64
    )
    offsets = torch.zeros(num_heads, dtype=torch.int64)
    offsets[1:] = sizes.cumsum(0)[:-1]
    embedding.layer_multipliers.copy_(multipliers)
    embedding.ngram_heads_vocab_sizes.copy_(sizes)
    embedding.ngram_heads_offsets.copy_(offsets)
    embedding.snapshot_host_hash_constants(max_batch_size=1)

    generator = torch.Generator().manual_seed(20260829)
    contexts = torch.randint(0, 50_000, (steps, 2), generator=generator)
    current_ids = torch.randint(0, 50_000, (steps,), generator=generator)
    contexts[::17, 0] = eos_token_id
    contexts[::29, 1] = eos_token_id
    multiplier_list = multipliers.tolist()
    size_list = sizes.tolist()
    offset_list = offsets.tolist()
    row_ids_ptr = None

    for step in range(steps):
        context = contexts[step : step + 1]
        current = current_ids[step : step + 1]
        reference_ids = _derive_decode_row_ids_host_reference(
            context.tolist(),
            current.tolist(),
            layer_multipliers=multiplier_list,
            vocab_sizes=size_list,
            offsets=offset_list,
            ngram_size=3,
            heads_per_ngram=heads_per_ngram,
            eos_token_id=eos_token_id,
        )
        row_ids = embedding.host_decode_row_ids(context, current)
        if row_ids_ptr is None:
            row_ids_ptr = row_ids.data_ptr()
        else:
            assert row_ids.data_ptr() == row_ids_ptr
        assert torch.equal(row_ids, reference_ids)
        reference_staged, reference_inverse = reference_backend._stage_rows_reference(
            reference_ids
        )
        fast_backend.prepare_decode(row_ids)
        unique_count = torch.unique(reference_ids).numel()
        assert torch.equal(
            fast_backend.local_ids[:1].long(), reference_inverse
        )
        assert torch.equal(
            fast_backend._stage_bank.tensor[:unique_count].view(torch.uint8),
            reference_staged.view(torch.uint8),
        )
        fast_backend.finish_decode(record_event=False)

    started = time.perf_counter()
    for step in range(steps):
        reference_ids = _derive_decode_row_ids_host_reference(
            contexts[step : step + 1].tolist(),
            current_ids[step : step + 1].tolist(),
            layer_multipliers=multiplier_list,
            vocab_sizes=size_list,
            offsets=offset_list,
            ngram_size=3,
            heads_per_ngram=heads_per_ngram,
            eos_token_id=eos_token_id,
        )
        reference_backend._stage_rows_reference(reference_ids)
    reference_rate = steps / (time.perf_counter() - started)

    started = time.perf_counter()
    for step in range(steps):
        row_ids = embedding.host_decode_row_ids(
            contexts[step : step + 1], current_ids[step : step + 1]
        )
        fast_backend.prepare_decode(row_ids)
        fast_backend.finish_decode(record_event=False)
    fast_rate = steps / (time.perf_counter() - started)
    print(
        f"PLE staging micro-benchmark: before={reference_rate:.1f} steps/sec, "
        f"after={fast_rate:.1f} steps/sec"
    )


@requires_cuda
@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_quantized_ple_pinned_kernel_matches_torch(tmp_path, table_format):
    from freetoken.models.qwen4_exp.ple import (
        DiskStagedTable,
        PinnedUVATable,
        dequantize_ple_rows,
    )

    _quantized_ple_checkpoint(tmp_path, table_format)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    table = load_ple_table(str(tmp_path), args, pin=False)
    table.bank.pin()
    assert table.scale_bank is not None
    table.scale_bank.pin()
    backend = PinnedUVATable(table, prefetch=False)
    ids = torch.tensor([[0, 9, 0], [table.num_rows - 1, 9, 3]], device="cuda")

    got = backend.lookup(ids).cpu()
    want = dequantize_ple_rows(
        table.tensor.index_select(0, ids.cpu().reshape(-1)),
        table.scales.index_select(0, ids.cpu().reshape(-1)),
        table.format,
        float(table.weight_scale),
    ).view_as(got)
    assert torch.equal(got, want)

    disk = load_ple_table(str(tmp_path), args, backend="disk")
    staged = DiskStagedTable(
        disk,
        stage_capacity_rows=8,
        device=torch.device("cuda"),
        prefetch=False,
        max_decode_batch_size=2,
        rows_per_token=3,
    )
    staged.prepare_decode(ids.cpu())
    staged_got = staged.lookup(torch.zeros_like(ids)).cpu()
    staged.finish_decode(record_event=False)
    assert torch.equal(staged_got, want)


@requires_cuda
def test_disk_ple_fixed_buffers_capture_and_replay_different_rows(checkpoint):
    """Two host-prepared row sets flow through one captured fixed-address UVA gather."""
    from freetoken.models.qwen4_exp.ple import DiskStagedTable

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    pinned = load_ple_table(folder, args, pin=False)
    disk = load_ple_table(folder, args, backend="disk")
    backend = DiskStagedTable(
        disk,
        stage_capacity_rows=8,
        device=torch.device("cuda"),
        prefetch=False,
        max_decode_batch_size=2,
        rows_per_token=3,
    )
    row_sets = [
        torch.tensor([[0, 3, 0], [8, 11, 8]], dtype=torch.int64),
        torch.tensor([[27, 1, 15], [1, 27, 4]], dtype=torch.int64),
    ]
    dummy_ids = torch.zeros((2, 3), dtype=torch.int64, device="cuda")

    backend.prepare_decode(row_sets[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.lookup(dummy_ids)
    backend.finish_decode(record_event=False)

    for ids in row_sets:
        backend.prepare_decode(ids)
        graph.replay()
        backend.finish_decode(record_event=True)
        torch.cuda.synchronize()
        got = captured.clone()
        raw = pinned.tensor.view(torch.uint8).index_select(0, ids.reshape(-1))
        want = (
            raw.view(torch.float8_e4m3fn).to(torch.bfloat16) * backend.scale
        ).view_as(got).to(got.device)
        assert torch.equal(got, want)


@requires_cuda
def test_hmm_startup_probe_and_cuda_graph_replay_match_pinned(checkpoint):
    """HMM keeps live ids and a valid padded dummy row across graph replay."""
    from freetoken.models.qwen4_exp.ple import HMMMappedTable, PinnedUVATable

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    pinned = load_ple_table(folder, args, pin=False)
    pinned.bank.pin()
    reference = PinnedUVATable(
        pinned.tensor, float(pinned.weight_scale), prefetch=False
    )
    mapped = load_ple_table(folder, args, backend="hmm")
    hmm = HMMMappedTable(mapped, prefetch=False)
    hmm.startup_probe()

    ids = torch.tensor(
        [[0, 6, 7], [27, 14, 1], [0, 0, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    assert torch.equal(hmm.lookup(ids), reference.lookup(ids))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hmm.lookup(ids)
    ids.copy_(
        torch.tensor([[26, 8, 3], [21, 13, 0], [0, 0, 0]], device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, reference.lookup(ids))


@requires_cuda
def test_hmm_prefill_gather_matches_direct_fault_path(checkpoint):
    from freetoken.models.qwen4_exp.ple import HMMMappedTable, PrefillGatherTable

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    mapped = load_ple_table(folder, args, backend="hmm")
    hmm = HMMMappedTable(mapped, prefetch=False)
    gather = PrefillGatherTable(
        hmm,
        mapped,
        max_prefill_tokens=3,
        rows_per_token=3,
        device=torch.device("cuda"),
    )
    ids_host = torch.tensor([[0, 6, 7], [27, 14, 1], [3, 3, 21]])
    ids_device = ids_host.cuda()
    direct = hmm.lookup(ids_device).clone()

    assert gather.prepare_prefill(ids_host)
    gather.prefetch(ids_device)
    staged = gather.lookup(ids_device)
    assert torch.equal(staged, direct)


def test_pinned_ple_backend_never_constructs_file_mapping(checkpoint, monkeypatch):
    import freetoken.models.qwen4_exp.weight as weight

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    real_host_bank = weight.HostBank
    backings = []

    def tracked_host_bank(*shape, **kwargs):
        backings.append(kwargs.get("backing"))
        return real_host_bank(*shape, **kwargs)

    monkeypatch.setattr(weight, "HostBank", tracked_host_bank)
    table = weight.load_ple_table(folder, args, backend="pinned", pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert "file" not in backings


@pytest.mark.parametrize("backend", ["pinned", "disk"])
def test_ple_banks_obey_off_policy_and_join_model_scope(
    checkpoint, backend, monkeypatch,
):
    import freetoken.models.qwen4_exp.weight as weight

    from freetoken.moe.host_banks import requested_hugepages

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    if backend == "pinned":
        monkeypatch.setattr(weight, "read_range_into", lambda *_args, **_kwargs: 0)
    with requested_hugepages("off") as scope:
        table = load_ple_table(folder, args, backend=backend, pin=False)

    expected = (
        {bank for bank in (table.bank, table.scale_bank) if bank is not None}
        if backend == "pinned"
        else set((*table.banks, *table.scale_banks))
    )
    assert expected <= set(scope.banks.values())
    assert {
        tensor._freetoken_host_bank for tensor in scope.sources["PLE table"]
    } == expected
    for bank in scope.banks.values():
        assert bank._hugepage_status["mode"] == "off"
        assert bank._hugepage_status["advised_bytes"] == 0


def test_disk_ple_banks_are_excluded_from_forced_file_thp(checkpoint, monkeypatch):
    from freetoken.moe import host_banks

    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    with host_banks.requested_hugepages("on"):
        table = load_ple_table(folder, args, backend="disk", pin=False)

    assert all(
        bank._hugepage_status["reason"].startswith(
            "excluded: file THP not available for PLE"
        )
        for bank in (*table.banks, *table.scale_banks)
    )


def test_disk_ple_stats_report_pages_and_major_fault_delta(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple

    class Backend:
        def __init__(self, pages):
            self.prefetch_pages = pages

        def reset_stats(self):
            self.prefetch_pages = 0

    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._ple_disk_backends = [Backend(7), Backend(11)]
    model._ple_major_fault_base = 20
    model._ple_staging_ns = 12_500
    monkeypatch.setattr(ple, "process_major_faults", lambda: 25)

    assert model.ple_disk_stats(reset=True) == {
        "ple_prefetch_pages": 18,
        "ple_major_faults": 5,
        "ple_staging_us": 12.5,
    }
    assert [backend.prefetch_pages for backend in model._ple_disk_backends] == [0, 0]
    assert model._ple_major_fault_base == 25
    assert model._ple_staging_ns == 0


def test_hmm_ple_stats_report_major_fault_delta_without_staging(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple

    backend = SimpleNamespace(
        prefetch_pages=0,
        prefill_gather_rows=41,
        prefill_gather_ms=7.25,
        reset_stats=lambda: None,
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._ple_hmm_backends = [backend]
    model._ple_major_fault_base = 40
    model._ple_staging_ns = 0
    monkeypatch.setattr(ple, "process_major_faults", lambda: 43)

    assert model.ple_disk_stats() == {
        "ple_prefetch_pages": 0,
        "ple_major_faults": 3,
        "ple_staging_us": 0,
        "ple_prefill_gather_rows": 41,
        "ple_prefill_gather_ms": 7.25,
    }


def test_cached_ple_stats_report_direct_hit_rate(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple

    class Backend:
        prefetch_pages = 3

        def cache_stats(self):
            return {
                "hits": 90,
                "misses": 10,
                "evictions": 4,
                "installed_rows": 7,
                "overflow_fallbacks": 1,
            }

        def reset_stats(self):
            pass

    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._ple_disk_backends = [Backend()]
    model._ple_major_fault_base = 10
    model._ple_staging_ns = 8_000
    monkeypatch.setattr(ple, "process_major_faults", lambda: 12)

    assert model.ple_disk_stats() == {
        "ple_prefetch_pages": 3,
        "ple_major_faults": 2,
        "ple_staging_us": 8.0,
        "ple_hits": 90,
        "ple_misses": 10,
        "ple_evictions": 4,
        "ple_installed_rows": 7,
        "ple_hit_rate": 0.9,
        "ple_overflow_fallbacks": 1,
    }


def test_disk_ple_prepare_replay_handles_mixed_history_abort_and_padding():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    class Event:
        def __init__(self):
            self.waits = 0

        def synchronize(self):
            self.waits += 1

    class Embedding:
        def __init__(self):
            self.seen = None

        def host_decode_row_ids(self, contexts, current_ids):
            self.seen = (contexts.clone(), current_ids.clone())
            return torch.zeros((contexts.shape[0], 16), dtype=torch.int64)

    class Backend:
        def __init__(self):
            self.ids = None

        def prepare_decode(self, ids):
            self.ids = ids

    done = Event()
    reqs = [
        SimpleNamespace(
            uid=1,
            cached_len=2,
            input_ids=torch.tensor([10, 11, 12]),
            pending_token_cpu=None,
            sample_copy_done=None,
            aborted=False,
        ),
        SimpleNamespace(
            uid=2,
            cached_len=2,
            input_ids=torch.tensor([20, 21]),
            pending_token_cpu=torch.tensor(22),
            sample_copy_done=done,
            aborted=False,
        ),
        SimpleNamespace(
            uid=3,
            cached_len=1,
            input_ids=torch.tensor([31]),
            pending_token_cpu=torch.tensor(32),
            sample_copy_done=done,
            # An abort received after launch still has to finish the in-flight row.
            aborted=True,
        ),
        SimpleNamespace(
            uid=-1,
            cached_len=0,
            input_ids=torch.tensor([0]),
            pending_token_cpu=None,
            sample_copy_done=None,
            aborted=False,
        ),
    ]
    embedding = Embedding()
    backend = Backend()
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(ngram_size=3, ngram_boundary_token_id=2)
    )
    model.model = SimpleNamespace(
        ple_layers=[SimpleNamespace(ple_embedding=embedding)]
    )
    model._ple_disk_backends = [backend]
    model._ple_decode_contexts = torch.empty((4, 2), dtype=torch.int64)
    model._ple_decode_input_ids = torch.empty(4, dtype=torch.int64)
    model._ple_waited_events = [None] * 4
    model._ple_staging_ns = 0

    model.prepare_cuda_graph_replay(SimpleNamespace(padded_reqs=reqs))

    assert done.waits == 1
    assert embedding.seen is not None
    assert torch.equal(
        embedding.seen[0], torch.tensor([[10, 11], [20, 21], [2, 31], [2, 2]])
    )
    assert torch.equal(embedding.seen[1], torch.tensor([12, 22, 32, 0]))
    assert backend.ids.shape == (4, 16)
    assert model._ple_staging_ns > 0


def test_hmm_prefill_prepare_uses_final_host_request_slices():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    seen = []
    staged = []

    class Embedding:
        def host_prefill_row_ids(self, reqs, max_tokens):
            seen.append((reqs, max_tokens))
            return torch.tensor([[1, 2], [2, 3]])

    backend = SimpleNamespace(
        max_prefill_tokens=7,
        prepare_prefill=lambda ids: staged.append(ids.clone()),
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._ple_prefill_gather = [
        (SimpleNamespace(ple_embedding=Embedding()), backend)
    ]
    reqs = [SimpleNamespace(input_ids=torch.tensor([10, 11]), cached_len=0)]
    batch = SimpleNamespace(is_prefill=True, reqs=reqs)

    model.prepare_prefill_ple(batch)

    assert seen == [(reqs, 7)]
    assert torch.equal(staged[0], torch.tensor([[1, 2], [2, 3]]))


def test_disk_ple_load_reserves_zero_expert_pin_budget(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple
    import freetoken.models.qwen4_exp.weight as weight

    attached = []
    snapshotted = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(
            attach_table=attached.append,
            snapshot_host_hash_constants=lambda max_batch_size: snapshotted.append(
                max_batch_size
            ),
        ),
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(ple_layers=[layer])
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(num_ngram_heads=16, ngram_size=3),
    )
    mapped = SimpleNamespace(num_rows=32, head_dim=4)
    staged = SimpleNamespace(prefetch_pages=0)
    selected = []
    staging_args = []

    def fake_load(model_path, args, *, backend):
        selected.append(backend)
        return mapped

    def fake_staged_table(table, capacity, **kwargs):
        staging_args.append((capacity, kwargs))
        return staged

    monkeypatch.setattr(weight, "load_ple_table", fake_load)
    monkeypatch.setattr(ple, "DiskStagedTable", fake_staged_table)
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="disk",
        use_dummy_weight=False,
        max_running_req=3,
        max_forward_len=2,
        cuda_graph_bs=[1, 8],
        cuda_graph_max_bs=6,
    )

    assert model.load_host_tables(engine_config) == 0
    assert selected == ["disk"]
    assert attached == [staged]
    assert staging_args == [
        (8 * 16, {"max_decode_batch_size": 8, "rows_per_token": 16})
    ]
    assert model._ple_decode_contexts.shape == (8, 2)
    assert model._ple_decode_input_ids.shape == (8,)
    assert snapshotted == [8]


def test_uring_ple_load_charges_pinned_staging_to_expert_budget(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple_uring as ple_uring

    attached = []
    snapshotted = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(
            attach_table=attached.append,
            snapshot_host_hash_constants=snapshotted.append,
        )
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(ple_layers=[layer])
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(num_ngram_heads=16, ngram_size=3)
    )
    source = SimpleNamespace(num_rows=1000)
    constructed = []

    class FakeUring:
        staging_nbytes = 987_654
        prefetch_pages = 0

        def __init__(self, table, staging_mib, queue_depth, **kwargs):
            constructed.append((table, staging_mib, queue_depth, kwargs))

        def startup_description(self):
            return "backend=uring, test"

    monkeypatch.setattr(ple_uring, "resolve_uring_source", lambda *_args: source)
    monkeypatch.setattr(ple_uring, "UringTable", FakeUring)
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="uring",
        ple_uring_staging_mib=64,
        ple_uring_queue_depth=32,
        use_dummy_weight=False,
        max_running_req=3,
        max_forward_len=5,
        cuda_graph_bs=[1, 8],
        cuda_graph_max_bs=6,
    )

    assert model.load_host_tables(engine_config) == 987_654
    assert attached == model._ple_disk_backends
    assert snapshotted == [8]
    assert constructed == [
        (
            source,
            64,
            32,
            {
                "max_decode_batch_size": 8,
                "rows_per_token": 16,
                "required_capacity_rows": 128,
            },
        )
    ]


def test_cached_ple_load_reserves_row_bank_and_applies_warm_profile(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple
    import freetoken.models.qwen4_exp.weight as weight

    attached = []
    snapshotted = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(
            attach_table=attached.append,
            snapshot_host_hash_constants=lambda size: snapshotted.append(size),
        )
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(ple_layers=[layer])
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(num_ngram_heads=4, ngram_size=3)
    )
    mapped = SimpleNamespace(num_rows=100, head_dim=4)
    constructed = []

    class FakeCache:
        cache_nbytes = 1234
        prefetch_pages = 0

        def __init__(self, table, capacity, source_capacity, **kwargs):
            constructed.append((table, capacity, source_capacity, kwargs))

        def warm(self, rows):
            assert rows == [9, 2]
            return len(rows)

    monkeypatch.setattr(weight, "load_ple_table", lambda *args, **kwargs: mapped)
    monkeypatch.setattr(ple, "CachedTable", FakeCache)
    monkeypatch.setattr(ple, "ple_cache_capacity_rows", lambda budget, table: 64)
    monkeypatch.setattr(ple, "load_ple_row_profile", lambda path, rows: [9, 2])
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="cached",
        ple_cache_gib=1.5,
        ple_cache_warm="/tmp/hot.json",
        ple_cache_profile_out="/tmp/out.json",
        use_dummy_weight=False,
        max_running_req=2,
        max_forward_len=5,
        cuda_graph_bs=[1, 3],
        cuda_graph_max_bs=2,
    )

    assert model.load_host_tables(engine_config) == 1234
    assert attached == model._ple_disk_backends
    assert snapshotted == [3]
    assert constructed == [
        (
            mapped,
            64,
            20,
            {
                "max_decode_batch_size": 3,
                "rows_per_token": 4,
                "collect_profile": True,
            },
        )
    ]
    assert model._ple_cache_profile_out == "/tmp/out.json"


def test_hmm_ple_load_reserves_zero_expert_pin_budget(monkeypatch):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple
    import freetoken.models.qwen4_exp.weight as weight

    attached = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(attach_table=attached.append),
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(ple_layers=[layer])
    model._config = SimpleNamespace(qwen4_args=SimpleNamespace())
    mapped = SimpleNamespace(num_rows=32, head_dim=4)
    probes = []
    selected = []

    class FakeHMM:
        prefetch_pages = 0

        def __init__(self, table):
            assert table is mapped

        def startup_probe(self):
            probes.append(True)

    def fake_load(model_path, args, *, backend):
        selected.append(backend)
        return mapped

    monkeypatch.setattr(weight, "load_ple_table", fake_load)
    monkeypatch.setattr(ple, "HMMMappedTable", FakeHMM)
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="hmm",
        use_dummy_weight=False,
    )

    assert model.load_host_tables(engine_config) == 0
    assert selected == ["hmm"]
    assert attached == model._ple_hmm_backends
    assert probes == [True]
    assert not hasattr(model, "_ple_disk_backends")


@pytest.mark.parametrize("setting, expect_gather", [("on", True), ("off", False)])
def test_hmm_ple_prefill_gather_flag_gates_overlay(
    monkeypatch, setting, expect_gather
):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    import freetoken.models.qwen4_exp.ple as ple
    import freetoken.models.qwen4_exp.weight as weight

    attached = []
    snapshots = []
    embedding = SimpleNamespace(
        attach_table=attached.append,
        snapshot_host_hash_constants=lambda: snapshots.append(True),
    )
    layer = SimpleNamespace(ple_embedding=embedding)
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.model = SimpleNamespace(ple_layers=[layer])
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(num_ngram_heads=16)
    )
    table = SimpleNamespace(num_rows=100, head_dim=4)
    mapped = SimpleNamespace(num_rows=100, head_dim=4, prefetch_pages=0)
    wrappers = []

    class FakeHMM:
        def __new__(cls, source):
            assert source is table
            return mapped

    class FakeGather:
        enabled = True
        staging_nbytes = 123

        def __init__(self, fallback, source, max_tokens, rows_per_token):
            wrappers.append((fallback, source, max_tokens, rows_per_token))

    mapped.startup_probe = lambda: None
    monkeypatch.setattr(weight, "load_ple_table", lambda *args, **kwargs: table)
    monkeypatch.setattr(ple, "HMMMappedTable", FakeHMM)
    monkeypatch.setattr(ple, "PrefillGatherTable", FakeGather)
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="hmm",
        ple_prefill_gather=setting,
        max_extend_tokens=32,
        use_dummy_weight=False,
    )

    reserved = model.load_host_tables(engine_config)

    assert bool(wrappers) is expect_gather
    assert bool(snapshots) is expect_gather
    assert reserved == (123 if expect_gather else 0)
    assert attached == model._ple_hmm_backends
    assert attached[0] is (model._ple_prefill_gather[0][1] if expect_gather else mapped)


def test_load_ple_table_rejects_a_shard_count_mismatch(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS + 1, ngram_head_dim=NGRAM_DIM)
    with pytest.raises(ValueError, match="shards 0"):
        load_ple_table(folder, args, pin=False)


# ======================================================================================
# read_range_into: the O_DIRECT byte-range read the PLE table load is built on
# ======================================================================================


@pytest.fixture(scope="module")
def blob(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(7).randbytes(5_000_003)
    path = tmp_path_factory.mktemp("blob") / "data.bin"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("file_offset, nbytes, dest_offset", [
    (1, 4095, 0),                 # sub-block, unaligned source
    (2239, 1_000_000, 0),         # the real checkpoint's header-end phase
    (4095, 4097, 1),              # straddles two block boundaries
    (4_999_000, 1003, 123_456),   # runs to EOF
])
def test_read_range_into_matches_the_file(blob, file_offset, nbytes, dest_offset):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    got = read_range_into(view, path, file_offset=file_offset, nbytes=nbytes,
                          dest_offset=dest_offset, chunk=1 << 20)
    assert got == nbytes
    assert bytes(view[dest_offset:dest_offset + nbytes]) == data[file_offset:file_offset + nbytes]


def test_read_range_into_is_chunk_and_thread_safe(blob):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    read_range_into(view, path, file_offset=2239, nbytes=4_000_000, dest_offset=1024,
                    workers=8, chunk=64 << 10)
    assert bytes(view[1024:1024 + 4_000_000]) == data[2239:2239 + 4_000_000]


def test_read_range_into_rejects_a_short_destination(blob):
    path, _data = blob
    bank = HostBank((1024,), torch.uint8)
    with pytest.raises(ValueError, match="destination holds"):
        read_range_into(bank.memoryview(), path, file_offset=0, nbytes=1 << 20)


# ======================================================================================
# AOT shape table
# ======================================================================================


def test_aot_entry_carries_the_checkpoint_geometry():
    entry = next(m for m in SUPPORTED_MODELS
                 if m.architecture == "Qwen4ExpForConditionalGeneration")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (2560, 640, 10)
    assert entry.kv_groups == ((2, 256),)
    rows = expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size)
    assert set(rows) == {"gate_up_packed", "gate_up_scale", "gate_up_global",
                         "down_packed", "down_scale", "down_global"}
    for name, nbytes in rows.items():
        assert nbytes % 16 == 0, name  # fused multi-bank copy only engages on 16B multiples


def test_every_registry_architecture_is_claimed_by_an_aot_entry():
    from freetoken.models.register import _MODEL_REGISTRY

    claimed = {m.architecture for m in SUPPORTED_MODELS}
    claimed |= {a for m in SUPPORTED_MODELS for a in m.arch_aliases}
    assert "Qwen4ExpForConditionalGeneration" in claimed
    assert set(_MODEL_REGISTRY) - claimed == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_fusion_pad_rides_the_tensor_device():
    """safetensors loads straight to cuda; a cpu-allocated pad row would break torch.cat."""
    from freetoken.models.qwen4_exp.weight import _try_fuse

    buf = {}
    down = torch.randn(320, 64, device="cuda", dtype=torch.bfloat16)
    inject = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert _try_fuse("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", down, buf) == ()
    key, fused = _try_fuse("model.layers.0.attn_hyper_connection.block_inject_weight.weight", inject, buf)
    assert fused.device.type == "cuda" and fused.shape[0] == 336
    assert torch.equal(fused[324:], torch.zeros(12, 64, device="cuda", dtype=torch.bfloat16))
