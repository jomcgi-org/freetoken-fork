"""CPU-only FTW conversion coverage for GLM's native NVFP4 expert layout."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file


H = 16
I = 16
E = 2
LAYERS = 3
FIRST_MOE = 1


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)


def _source_checkpoint(path) -> dict[str, torch.Tensor]:
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "hidden_size": H,
        "vocab_size": 32,
        "intermediate_size": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "max_position_embeddings": 128,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "index_n_heads": 0,
        "index_head_dim": 0,
        "index_topk": 0,
        "indexer_types": [],
        "n_routed_experts": E,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": I,
        "first_k_dense_replace": FIRST_MOE,
        "n_shared_experts": 1,
        "quantization_config": {"quant_algo": "NVFP4"},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    torch.manual_seed(17)
    raw: dict[str, torch.Tensor] = {}
    for layer in range(LAYERS):
        attn = f"model.layers.{layer}.self_attn"
        raw.update({
            f"{attn}.q_a_proj.weight": _bf16(8, H),
            f"{attn}.q_b_proj.weight": _bf16(32, 8),
            f"{attn}.kv_a_proj_with_mqa.weight": _bf16(16, H),
            f"{attn}.kv_b_proj.weight": _bf16(48, 8),
            f"{attn}.o_proj.weight": _bf16(H, 32),
            f"{attn}.q_a_layernorm.weight": _bf16(8),
            f"{attn}.kv_a_layernorm.weight": _bf16(8),
            f"model.layers.{layer}.input_layernorm.weight": _bf16(H),
            f"model.layers.{layer}.post_attention_layernorm.weight": _bf16(H),
        })
        mlp = f"model.layers.{layer}.mlp"
        if layer < FIRST_MOE:
            raw.update({
                f"{mlp}.gate_proj.weight": _bf16(32, H),
                f"{mlp}.up_proj.weight": _bf16(32, H),
                f"{mlp}.down_proj.weight": _bf16(H, 32),
            })
            continue

        raw[f"{mlp}.gate.weight"] = _bf16(E, H)
        raw[f"{mlp}.gate.e_score_correction_bias"] = _bf16(E)
        for proj, out_features, in_features in (
            ("gate_proj", I, H),
            ("up_proj", I, H),
            ("down_proj", H, I),
        ):
            raw[f"{mlp}.shared_experts.{proj}.weight"] = _bf16(
                out_features, in_features
            )
        for expert in range(E):
            for proj, out_features, in_features in (
                ("gate_proj", I, H),
                ("up_proj", I, H),
                ("down_proj", H, I),
            ):
                prefix = f"{mlp}.experts.{expert}.{proj}"
                raw[f"{prefix}.weight"] = torch.randint(
                    0, 256, (out_features, in_features // 2), dtype=torch.uint8
                )
                raw[f"{prefix}.weight_scale"] = (
                    torch.arange(out_features * (in_features // 16), dtype=torch.float32)
                    .add_(1 + layer + expert)
                    .remainder_(32)
                    .view(out_features, in_features // 16)
                    .to(torch.float8_e4m3fn)
                )
                raw[f"{prefix}.weight_scale_2"] = torch.tensor(
                    0.25 * (1 + layer + expert), dtype=torch.float32
                )

    raw.update({
        "model.embed_tokens.weight": _bf16(32, H),
        "model.norm.weight": _bf16(H),
        "lm_head.weight": _bf16(32, H),
    })
    shard = "model-00001-of-00001.safetensors"
    save_file(raw, str(path / shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in raw}}), encoding="utf-8"
    )
    return raw


def _assert_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def test_glm_moe_dsa_cpu_ftw_round_trip_native_nvfp4(tmp_path, monkeypatch):
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.ftw import iter_ftw_weights, load_ftw_banks
    from freetoken.models.glm_moe_dsa import config as glm_config

    # Keep resident tensors checkpoint-native in this converter test. FP8 resident
    # requantization is covered by the model weight tests and is unrelated to expert banks.
    monkeypatch.setattr(glm_config, "_ATTN_FP8", False)
    monkeypatch.setattr(glm_config, "_MLP_FP8", False)

    source = tmp_path / "source"
    output = tmp_path / "ftw"
    source.mkdir()
    raw = _source_checkpoint(source)
    index = convert_checkpoint(
        str(source), str(output), moe_backend="offload", device="cpu",
        shard_limit=1 << 20,
    )

    assert index["quant_format"] == "nvfp4"
    assert index["expert_bank_geometry"] == {
        "num_layers": LAYERS - FIRST_MOE,
        "num_experts": E,
        "hidden_size": H,
        "moe_intermediate_size": I,
    }
    expected_layer_bytes = E * ((27 * H * I) // 16 + 4 * I + 2 * H)
    assert index["expert_bank_max_layer_bytes"] == expected_layer_bytes
    bank_entries = [entry for entry in index["tensors"] if entry["kind"] == "experts_bank"]
    assert len(bank_entries) == 6 * (LAYERS - FIRST_MOE)
    assert all("#L" in entry["name"] for entry in bank_entries)

    resident_entries = [entry for entry in index["tensors"] if entry["kind"] == "weight"]
    resident_names = {entry["name"] for entry in resident_entries}
    shared_names = {name for name in raw if ".mlp.shared_experts." in name}
    routed_names = {name for name in raw if ".mlp.experts." in name}
    assert shared_names <= resident_names
    assert routed_names.isdisjoint(resident_names)
    assert all("shared_experts" not in entry["name"] for entry in bank_entries)

    resident = {name: tensor.clone() for name, tensor in iter_ftw_weights(str(output))}
    for name in shared_names:
        assert torch.equal(resident[name], raw[name])

    banks = load_ftw_banks(
        str(output), num_layers=LAYERS - FIRST_MOE,
        layer_residency=["pageable"] * (LAYERS - FIRST_MOE),
    )
    assert banks is not None
    assert set(banks.sources) == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    for bank_layer, source_layer in enumerate(range(FIRST_MOE, LAYERS)):
        for expert in range(E):
            base = f"model.layers.{source_layer}.mlp.experts.{expert}"
            gate = raw[f"{base}.gate_proj.weight"]
            up = raw[f"{base}.up_proj.weight"]
            down = raw[f"{base}.down_proj.weight"]
            _assert_bits_equal(
                banks.sources["gate_up_packed"][bank_layer][expert],
                torch.cat((gate, up), dim=0),
            )
            _assert_bits_equal(
                banks.sources["down_packed"][bank_layer][expert], down
            )
            gate_scale = raw[f"{base}.gate_proj.weight_scale"]
            up_scale = raw[f"{base}.up_proj.weight_scale"]
            down_scale = raw[f"{base}.down_proj.weight_scale"]
            _assert_bits_equal(
                banks.sources["gate_up_scale"][bank_layer][expert],
                torch.cat((gate_scale, up_scale), dim=0),
            )
            _assert_bits_equal(
                banks.sources["down_scale"][bank_layer][expert], down_scale
            )
            gate_global = raw[f"{base}.gate_proj.weight_scale_2"].to(torch.float16)
            up_global = raw[f"{base}.up_proj.weight_scale_2"].to(torch.float16)
            down_global = raw[f"{base}.down_proj.weight_scale_2"].to(torch.float16)
            assert torch.equal(
                banks.sources["gate_up_global"][bank_layer][expert],
                torch.cat((gate_global.expand(I), up_global.expand(I))),
            )
            assert torch.equal(
                banks.sources["down_global"][bank_layer][expert],
                down_global.expand(H),
            )


def test_checkpoint_cli_without_gpu_never_initializes_cuda(tmp_path, monkeypatch):
    from freetoken.checkpoint import __main__ as checkpoint_main

    calls = []

    def fail_gpu(*_args, **_kwargs):
        raise AssertionError("CPU-only conversion must not initialize CUDA")

    def fake_convert(_model, _out, **kwargs):
        calls.append(kwargs)
        return {
            "counts": {"weight": 1, "mtp": 0, "experts_bank": 0},
            "total_bytes": 0,
            "shards": [],
            "quant_format": None,
            "fingerprint": None,
            "mtp_quant": None,
        }

    monkeypatch.setattr(checkpoint_main, "assign_gpu", fail_gpu)
    monkeypatch.setattr(checkpoint_main, "bind_assigned_gpu", fail_gpu)
    monkeypatch.setattr(checkpoint_main, "convert_checkpoint", fake_convert)
    assert checkpoint_main.main([
        "--model", str(tmp_path / "source"),
        "--out", str(tmp_path / "out"),
    ]) == 0
    assert calls[0]["device"] == "cpu"
