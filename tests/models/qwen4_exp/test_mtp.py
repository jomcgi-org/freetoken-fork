from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams, select_lm_head_rows
from freetoken.spec_decode import (
    MTP_DRAFT_STEPS,
    greedy_accept_prefix,
    validate_mtp_draft_tokens,
    validate_speculative_mtp,
)

from .common import parsed_config


def test_mtp_config_defaults_off_and_accepts_on():
    assert validate_speculative_mtp("off") == "off"
    assert validate_speculative_mtp("on") == "on"


def test_mtp_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="--speculative-mtp.*off.*on"):
        validate_speculative_mtp("auto")


def test_mtp_draft_tokens_is_fixed_at_one():
    assert MTP_DRAFT_STEPS == 1
    assert validate_mtp_draft_tokens(1) == 1
    with pytest.raises(ValueError, match=r"--mtp-draft-tokens.*fixed at 1"):
        validate_mtp_draft_tokens(2)


@pytest.mark.parametrize(
    ("drafts", "targets", "accepted", "matched"),
    [
        ([4, 5, 6], [4, 5, 6, 7], [4, 5, 6, 7], 3),
        ([4, 5, 6], [4, 9, 8, 7], [4, 9], 1),
        ([4, 5, 6], [9, 8, 7, 6], [9], 0),
        ([], [9], [9], 0),
    ],
)
def test_greedy_accept_prefix(drafts, targets, accepted, matched):
    got, got_matched = greedy_accept_prefix(torch.tensor(drafts), torch.tensor(targets))
    assert got.tolist() == accepted
    assert got_matched == matched


def test_greedy_accept_requires_bonus_prediction():
    with pytest.raises(ValueError, match=r"len\(drafts\) \+ 1"):
        greedy_accept_prefix(torch.tensor([1, 2]), torch.tensor([1, 2]))


@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.1])
def test_temperature_zero_is_greedy_regardless_of_top_p(top_p):
    assert SamplingParams(temperature=0.0, top_p=top_p).is_greedy
    assert SamplingParams(temperature=-1.0, top_p=top_p).is_greedy


def test_positive_temperature_keeps_existing_greedy_gate():
    assert SamplingParams(temperature=0.7, top_k=1, top_p=1.0).is_greedy
    assert not SamplingParams(temperature=0.7, top_k=-1, top_p=1.0).is_greedy
    assert not SamplingParams(temperature=0.7, top_k=1, top_p=0.9).is_greedy


def test_mtp_multi_position_lm_head_row_selection():
    """Verify and one-row draft projections must bypass prefill's last-row gather."""
    width, hidden_size = 2, 3
    last = torch.tensor([width - 1])
    batch = SimpleNamespace(
        is_prefill=True,
        size=1,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]),
    )
    verify_hidden = torch.arange(width * hidden_size).view(width, hidden_size)

    # Ordinary prefill projects only the final row. MTP verification requests all
    # positions so greedy_accept_prefix receives seed and draft predictions.
    assert torch.equal(select_lm_head_rows(verify_hidden, batch), verify_hidden[-1:])
    assert select_lm_head_rows(
        verify_hidden, batch, select_last=False
    ).shape == (width, hidden_size)

    # Each MTP draft call owns only one hidden row. Applying request-level
    # multi-position metadata here was the CUDA IndexKernel out-of-bounds crash.
    draft_hidden = verify_hidden[:1]
    with pytest.raises(IndexError):
        select_lm_head_rows(draft_hidden, batch)
    assert select_lm_head_rows(
        draft_hidden, batch, select_last=False
    ).data_ptr() == draft_hidden.data_ptr()


@pytest.mark.parametrize("mtp_quant", ["bf16", "nvfp4"])
def test_synthetic_ftw_records_mtp_quant_and_tensor_layout(tmp_path, mtp_quant):
    from freetoken.checkpoint.convert import (
        iter_converted_mtp_weights,
        mtp_ftw_metadata,
    )
    from freetoken.checkpoint.ftw import (
        FTWReader,
        FTWWriter,
        mtp_quant_from_checkpoint,
    )

    weights = [
        ("mtp.fc_embedding.weight", torch.ones(16, 16, dtype=torch.bfloat16)),
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            torch.randn(2, 32, 16, dtype=torch.bfloat16),
        ),
        (
            "mtp.layers.0.mlp.experts.down_proj",
            torch.randn(2, 16, 16, dtype=torch.bfloat16),
        ),
    ]
    converted = list(iter_converted_mtp_weights(iter(weights), mtp_quant))
    writer = FTWWriter(str(tmp_path))
    expert_bytes = 0
    for name, tensor in converted:
        writer.add_tensor(name, tensor, kind="mtp")
        if ".mlp.experts." in name:
            expert_bytes += tensor.numel() * tensor.element_size()
    writer.finalize(
        {
            **mtp_ftw_metadata(mtp_quant, len(converted), expert_bytes),
            "counts": {"weight": 0, "mtp": len(converted), "experts_bank": 0},
        }
    )

    reader = FTWReader(str(tmp_path))
    try:
        assert reader.meta("mtp_quant") == mtp_quant
        assert reader.meta("mtp_expert_bytes") == expert_bytes
        entries = {entry["name"]: entry for entry in reader.entries("mtp")}
    finally:
        reader.close()
    if mtp_quant == "bf16":
        assert entries["mtp.layers.0.mlp.experts.gate_up_proj"]["dtype"] == "bfloat16"
        assert len(entries) == 3
    else:
        assert entries["mtp.layers.0.mlp.experts.gate_up_packed"]["dtype"] == "uint8"
        assert entries["mtp.layers.0.mlp.experts.gate_up_scale"]["dtype"] == "float8_e4m3fn"
        assert entries["mtp.layers.0.mlp.experts.gate_up_global"]["dtype"] == "float16"
        assert len(entries) == 7
    assert mtp_quant_from_checkpoint(str(tmp_path)) == mtp_quant


def test_nvfp4_group16_shape_dtype_and_lossless_construction():
    from freetoken.models.nvfp4_banks import (
        dequantize_nvfp4_group16,
        quantize_nvfp4_group16,
    )

    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0]
    )
    # A scale of 448 is exactly representable by e4m3 with an fp16 global of one.
    weight = (levels * 448).to(torch.bfloat16).repeat(2, 3, 1)
    packed, scale, row_global = quantize_nvfp4_group16(weight, row_chunk=2)
    assert packed.shape == (2, 3, 8) and packed.dtype == torch.uint8
    assert scale.shape == (2, 3, 1) and scale.dtype == torch.float8_e4m3fn
    assert row_global.shape == (2, 3) and row_global.dtype == torch.float16
    assert torch.equal(dequantize_nvfp4_group16(packed, scale, row_global), weight)


@pytest.mark.parametrize("mtp_quant", ["bf16", "nvfp4"])
def test_mtp_experts_are_resident_and_never_use_offload_cache(mtp_quant):
    from freetoken.layers.moe import MoELayer, OffloadMoELayer
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPMoE
    from freetoken.utils.torch_utils import torch_dtype

    config = parsed_config(hidden=256, moe_intermediate_size=128)
    # The target is offloaded, but the separately-built head must ignore that backend.
    object.__setattr__(config, "moe_backend", "offload")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        mtp_moe = Qwen4ExpMTPMoE(config, mtp_quant)
    assert type(mtp_moe.experts) is MoELayer
    assert not isinstance(mtp_moe.experts, OffloadMoELayer)
    assert mtp_moe.experts.weight_format == mtp_quant
    assert not hasattr(mtp_moe.experts, "offload_cache")
    expert_keys = set(mtp_moe.experts.state_dict())
    if mtp_quant == "bf16":
        assert expert_keys == {"gate_up_proj", "down_proj"}
    else:
        assert expert_keys == {
            "gate_up_packed",
            "gate_up_scale",
            "gate_up_global",
            "down_packed",
            "down_scale",
            "down_global",
        }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_lossless_nvfp4_mtp_logits_match_reference_dequant():
    from freetoken.core import Context, set_global_ctx
    from freetoken.layers.moe import MoELayer
    from freetoken.models.nvfp4_banks import (
        dequantize_nvfp4_group16,
        quantize_nvfp4_group16,
    )

    device = torch.device("cuda")
    e, h, i = 2, 256, 128
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0]
    )

    def lossless(*shape):
        count = math.prod(shape)
        return (levels.repeat((count + 15) // 16)[:count] * 448).reshape(shape).to(
            torch.bfloat16
        )

    gate_up = lossless(e, 2 * i, h)
    down = lossless(e, h, i)
    gu_p, gu_s, gu_g = quantize_nvfp4_group16(gate_up)
    dn_p, dn_s, dn_g = quantize_nvfp4_group16(down)
    assert torch.equal(dequantize_nvfp4_group16(gu_p, gu_s, gu_g), gate_up)
    assert torch.equal(dequantize_nvfp4_group16(dn_p, dn_s, dn_g), down)

    with torch.device(device):
        quant = MoELayer(e, 1, h, i, weight_format="nvfp4")
        reference = MoELayer(e, 1, h, i, weight_format="bf16")
    quant.gate_up_packed = gu_p.to(device)
    quant.gate_up_scale = gu_s.to(device)
    quant.gate_up_global = gu_g.to(device)
    quant.down_packed = dn_p.to(device)
    quant.down_scale = dn_s.to(device)
    quant.down_global = dn_g.to(device)
    reference.gate_up_proj = gate_up.to(device)
    reference.down_proj = down.to(device)

    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    hidden = torch.randn(1, h, dtype=torch.bfloat16, device=device) / 448
    topk_weights = torch.ones(1, 1, dtype=torch.float32, device=device)
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device=device)
    with ctx.forward_batch(SimpleNamespace(is_prefill=False)):
        ref_hidden = reference.routed_forward(hidden, topk_weights, topk_ids)
        got_hidden = quant.routed_forward(hidden, topk_weights, topk_ids)

    # Synthetic lm_head chosen from the reference row makes token zero a robust greedy
    # winner while still comparing the complete quantized draft-logit plumbing.
    lm_weight = torch.zeros(4, h, dtype=torch.float32, device=device)
    lm_weight[0] = ref_hidden[0].float()
    lm_weight[1] = -ref_hidden[0].float()
    ref_logits = torch.nn.functional.linear(ref_hidden.float(), lm_weight)
    got_logits = torch.nn.functional.linear(got_hidden.float(), lm_weight)
    tol = 0.03 * float(ref_logits.abs().max())
    torch.testing.assert_close(got_logits, ref_logits, rtol=3e-2, atol=tol)
    assert got_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_tiny_model_speculation_matches_greedy():
    class TinyTarget(torch.nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            transition = torch.arange(vocab_size).roll(-1)
            self.register_buffer("transition", transition)
            self.register_buffer("state", torch.zeros(4, dtype=torch.int64))
            self.vocab_size = vocab_size

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            rows = []
            for token in tokens.reshape(-1):
                self.state.mul_(31).add_(token.to(torch.int64) + 1)
                logits = torch.full(
                    (self.vocab_size,), -1000.0, device=tokens.device
                )
                logits[self.transition[token]] = 0.0
                rows.append(logits)
            return torch.stack(rows)

    device = torch.device("cuda")
    greedy_model = TinyTarget(vocab_size=17).to(device)
    speculative_model = TinyTarget(vocab_size=17).to(device)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
    assert sampling_params.is_greedy
    greedy = [3]
    speculative = [3]
    for window in range(12):
        assert sampling_params.is_greedy  # the scheduler's MTP engagement gate
        seed = torch.tensor(speculative[-1], device=device)
        correct = (seed + 1) % 17
        draft = correct if window % 3 else (seed + 4) % 17

        snapshot = speculative_model.state.clone()
        targets = speculative_model(torch.stack((seed, draft))).argmax(dim=-1)
        accepted, matched = greedy_accept_prefix(draft.view(1), targets)
        if matched == 0:
            speculative_model.state.copy_(snapshot)
            accepted = speculative_model(seed.view(1)).argmax(dim=-1)
        speculative.extend(accepted.tolist())

        greedy.append(
            int(greedy_model(torch.tensor([greedy[-1]], device=device)).argmax())
        )
        if matched:
            greedy.append(
                int(greedy_model(torch.tensor([greedy[-1]], device=device)).argmax())
            )
        assert speculative == greedy
        assert torch.equal(speculative_model.state, greedy_model.state)
