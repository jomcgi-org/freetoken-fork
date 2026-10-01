"""``--dense-weight-dtype fp8`` / ``--fp8-lm-head`` for Qwen3.8-Flash-Next (GPU-free).

Flag parsing and validation, the lm_head / MTP rule, which tensors the module swap converts (the
meta-device toy model and the real checkpoint's tensor list are held to the same pattern table),
the streaming load-time quantizer, the byte accounting against the shipped checkpoint, and the
fp8 hyper-connection block against its BF16 original on CPU.
"""

from __future__ import annotations

import json
import os
import re

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.kernel.triton.fp8_dense_linear import (
    FP8,
    Fp8DenseLinear,
    Fp8DenseLMHead,
    quantize_rowwise_fp8,
)
from freetoken.models.dense_fp8 import (
    DENSE_FP8_PATTERNS,
    LM_HEAD_KEY,
    DenseFp8Policy,
    apply_dense_fp8,
    is_dense_fp8_key,
    projection_shapes,
    quantize_loaded_weights,
    resolve_dense_fp8_policy,
    scale_key,
)
from freetoken.models.qwen4_exp.hc import GatedResidual
from freetoken.server.args import parse_args
from freetoken.utils.torch_utils import torch_dtype

from .common import toy_hf_config
from .test_skeleton import _config, _fill

MODEL_DIR = "/var/lib/longhorn/nvme-02/freetoken/models/flash-e2m1.ftw"


# --------------------------------------------------------------------------------------
# flags
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype,head,mtp,dense,lm_head",
    [
        ("bf16", None, "off", False, False),
        ("bf16", None, "on", False, False),
        ("bf16", "off", "off", False, False),
        ("fp8", None, "off", True, True),  # default: lm_head follows fp8
        ("fp8", "on", "off", True, True),
        ("fp8", "off", "off", True, False),
        ("fp8", None, "on", True, False),  # default under MTP: draft shares lm_head -> bf16
        ("fp8", "off", "on", True, False),
    ],
)
def test_policy_resolution(dtype, head, mtp, dense, lm_head):
    assert resolve_dense_fp8_policy(dtype, head, mtp) == DenseFp8Policy(dense, lm_head)


@pytest.mark.parametrize(
    "dtype,head,mtp,match",
    [
        ("fp16", None, "off", "--dense-weight-dtype"),
        ("fp8", "auto", "off", "--fp8-lm-head"),
        ("bf16", "on", "off", "requires --dense-weight-dtype fp8"),
        ("fp8", "on", "on", "shares the target's lm_head"),
    ],
)
def test_policy_rejections(dtype, head, mtp, match):
    with pytest.raises(ValueError, match=match):
        resolve_dense_fp8_policy(dtype, head, mtp)


def test_engine_config_defaults_are_unchanged_and_validated():
    kwargs = dict(model_path="/tmp/model", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    config = EngineConfig(**kwargs)
    assert config.dense_weight_dtype == "bf16" and config.fp8_lm_head is None
    EngineConfig(**kwargs, dense_weight_dtype="fp8")
    with pytest.raises(ValueError, match="--dense-weight-dtype"):
        EngineConfig(**kwargs, dense_weight_dtype="int8")
    with pytest.raises(ValueError, match="shares the target's lm_head"):
        EngineConfig(**kwargs, dense_weight_dtype="fp8", fp8_lm_head="on", speculative_mtp="on")


def test_cli_flags_parse():
    model = ["--model", "/models/anon", "--dtype", "bfloat16"]
    args, _ = parse_args(model)
    assert (args.dense_weight_dtype, args.fp8_lm_head) == ("bf16", None)
    args, _ = parse_args([*model, "--dense-weight-dtype", "fp8"])
    assert (args.dense_weight_dtype, args.fp8_lm_head) == ("fp8", None)
    args, _ = parse_args([*model, "--dense-weight-dtype", "fp8", "--fp8-lm-head", "off"])
    assert args.fp8_lm_head == "off"
    args, _ = parse_args([*model, "--dense-weight-dtype", "fp8", "--speculative-mtp", "on"])
    assert resolve_dense_fp8_policy(
        args.dense_weight_dtype, args.fp8_lm_head, args.speculative_mtp
    ) == DenseFp8Policy(True, False)
    with pytest.raises(SystemExit):
        parse_args([*model, "--dense-weight-dtype", "int4"])
    with pytest.raises(ValueError, match="shares the target's lm_head"):
        parse_args([
            *model, "--dense-weight-dtype", "fp8", "--fp8-lm-head", "on", "--speculative-mtp", "on",
        ])


# --------------------------------------------------------------------------------------
# which tensors are converted
# --------------------------------------------------------------------------------------
_KEPT_BF16 = [
    "model.embed_tokens.weight",
    "model.layers.3.mlp.gate.weight",
    "model.layers.3.mlp.shared_expert_gate.weight",
    "model.layers.3.self_attn.indexer.index_qk_proj.weight",
    "model.layers.3.self_attn.q_norm.weight",
    "model.layers.3.attn_hyper_connection.hc_norm.weight",
    "model.layers.1.ple.key_proj.weight",
    "model.layers.1.ple.value_proj.weight",
    "model.layers.0.linear_attn.conv1d.weight",
    "model.layers.0.linear_attn.A_log",
    "mtp.layers.0.self_attn.qkv_proj.weight",
    "mtp.fc_hidden.weight",
    "lm_head.weight",  # governed by its own flag, not the pattern table
]


@pytest.mark.parametrize("key", _KEPT_BF16)
def test_pattern_table_keeps_small_and_sensitive_weights_bf16(key):
    assert not is_dense_fp8_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.3.self_attn.qkv_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.0.linear_attn.in_proj.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.47.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.47.mlp.shared_expert.down_proj.weight",
        "model.layers.5.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "model.layers.5.mlp_hyper_connection.input_mix_weight_up.weight",
        "model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.hyper_connection_mixer.input_mix_weight_up.weight",
    ],
)
def test_pattern_table_matches_dense_projections(key):
    assert is_dense_fp8_key(key)


def _build(mtp: bool = False):
    from freetoken.layers import set_rope_device

    set_rope_device(torch.device("cpu"))  # the engine does this before building on meta
    config = _config()
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

        model = Qwen4ExpForCausalLM(config)
        if mtp:
            model.enable_mtp("bf16")
    return config, model


def test_apply_converts_exactly_the_pattern_keys():
    _, model = _build()
    before = {k: (v.shape, v.dtype) for k, v in model.state_dict().items()}
    expect = {k for k in before if is_dense_fp8_key(k)}
    assert expect  # the toy model has attention, GDN, shared expert and HC projections

    report = apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=False))
    after = model.state_dict()
    assert set(report.keys) == expect and not report.lm_head
    for key in expect:
        assert after[key].dtype == FP8 and after[key].shape == before[key][0]
        assert after[scale_key(key)].dtype == torch.float32
        assert after[scale_key(key)].shape == (before[key][0][0],)
    # everything else is byte-for-byte the old layout
    for key, (shape, dtype) in before.items():
        if key not in expect:
            assert after[key].shape == shape and after[key].dtype == dtype
    assert set(after) == set(before) | {scale_key(k) for k in expect}
    assert isinstance(model.lm_head, type(_build()[1].lm_head))  # lm_head stayed BF16
    # accounting: bf16 2 B/elem -> fp8 1 B/elem + 4 B per row
    assert report.bf16_bytes == sum(2 * before[k][0].numel() for k in expect)
    assert report.fp8_bytes == sum(
        before[k][0].numel() + 4 * before[k][0][0] for k in expect
    )
    assert report.freed_bytes > 0 and "freed" in report.summary()


def test_apply_with_lm_head_and_bf16_policy_noop():
    _, model = _build()
    n_before = len(model.state_dict())
    noop = apply_dense_fp8(model, DenseFp8Policy(dense=False, lm_head=False))
    assert noop.keys == [] and len(model.state_dict()) == n_before

    report = apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=True))
    assert report.lm_head and LM_HEAD_KEY in report.keys
    assert isinstance(model.lm_head, Fp8DenseLMHead)
    state = model.state_dict()
    assert state["lm_head.weight"].dtype == FP8 and state["lm_head.weight_scale"].dtype == torch.float32


def test_mtp_head_stays_bf16_and_keeps_its_own_layers():
    _, model = _build(mtp=True)
    mtp_before = {k: v.dtype for k, v in model.state_dict().items() if k.startswith("mtp.")}
    assert mtp_before
    apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=False))
    state = model.state_dict()
    assert {k: v.dtype for k, v in state.items() if k.startswith("mtp.")} == mtp_before
    assert not any(k.startswith("mtp.") and k.endswith("weight_scale") for k in state)
    mtp_layers = [type(m).__name__ for m in model.mtp.layers.op_list[0].self_attn.__dict__.values()]
    assert "Fp8DenseLinear" not in mtp_layers
    # the draft path projects through model.lm_head: BF16 under the default MTP policy
    policy = resolve_dense_fp8_policy("fp8", None, "on")
    assert not policy.lm_head


def test_unmatched_linears_are_never_swapped():
    _, model = _build()
    apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=False))
    layer = model.model.layers.op_list[3]  # full-attention layer
    assert isinstance(layer.self_attn.qkv_proj, Fp8DenseLinear)
    assert isinstance(layer.self_attn.o_proj, Fp8DenseLinear)
    assert not isinstance(layer.self_attn.indexer.index_qk_proj, Fp8DenseLinear)
    assert not isinstance(layer.mlp.gate, Fp8DenseLinear)
    assert not isinstance(layer.mlp.shared_expert_gate, Fp8DenseLinear)
    assert isinstance(layer.mlp.shared_expert.down_proj, Fp8DenseLinear)  # LinearRowParallel


# --------------------------------------------------------------------------------------
# load-time quantization
# --------------------------------------------------------------------------------------
def test_quantize_loaded_weights_streams_and_matches_model_layout():
    _, model = _build()
    apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=True))
    state = model.state_dict()
    g = torch.Generator().manual_seed(0)
    originals = {}

    def checkpoint():  # a BF16 checkpoint: what the stream sees
        for key, param in state.items():
            if key.endswith("weight_scale"):
                continue
            if param.dtype == FP8:
                w = (torch.randn(param.shape, generator=g) * 0.05).to(torch.bfloat16)
                originals[key] = w
                yield key, w
            elif param.dtype.is_floating_point:
                yield key, torch.zeros(param.shape, dtype=param.dtype)
            else:
                yield key, torch.zeros(param.shape, dtype=param.dtype)

    produced = dict(quantize_loaded_weights(checkpoint(), state, torch.device("cpu")))
    assert set(produced) == set(state)  # exactly the model's keys: scales appear, nothing else
    for key, param in state.items():
        assert produced[key].dtype == param.dtype and produced[key].shape == param.shape, key
    for key, w in originals.items():
        q, scale = quantize_rowwise_fp8(w)
        assert torch.equal(produced[key].view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(produced[scale_key(key)], scale)
    # and the model accepts it
    model.load_state_dict(dict(produced))


def test_quantize_loaded_weights_is_lazy_and_passes_fp8_through():
    _, model = _build()
    apply_dense_fp8(model, DenseFp8Policy(dense=True, lm_head=False))
    state = model.state_dict()
    key = next(k for k in state if is_dense_fp8_key(k))
    pulled = []

    def source():
        for i in range(3):
            pulled.append(i)
            yield f"unrelated.{i}", torch.zeros(1)
        pulled.append("fp8")
        yield key, torch.zeros(state[key].shape, dtype=FP8)

    stream = quantize_loaded_weights(source(), state, torch.device("cpu"))
    assert next(stream)[0] == "unrelated.0" and pulled == [0]  # one tensor at a time
    rest = dict(stream)
    assert key in rest and scale_key(key) not in rest  # already fp8: scale must be the checkpoint's


# --------------------------------------------------------------------------------------
# numerics of a converted block
# --------------------------------------------------------------------------------------
def test_fp8_hyper_connection_matches_bf16_block_on_cpu():
    config = _config()
    gen = torch.Generator().manual_seed(11)
    with torch_dtype(torch.float32):
        hc = GatedResidual(config)
    _fill(hc, gen, 0.05)
    hc.hc_norm.weight.zero_()
    R = torch.randn(5, hc.hc_count * hc.hidden_size, generator=gen)
    x_ref, s_ref = hc.mix(R)

    class _Holder:  # apply_dense_fp8 walks model.model; give it a one-block tree
        pass

    holder = _Holder()
    holder.model = hc
    holder.lm_head = None
    # the pattern table is anchored at "model.", so run the swap by hand for the block's names
    for attr in ("input_mix_weight_down_block_inject", "input_mix_weight_up"):
        old = getattr(hc, attr)
        q, scale = quantize_rowwise_fp8(old.weight)
        new = Fp8DenseLinear(old.weight.shape[1], old.weight.shape[0])
        new.weight, new.weight_scale = q, scale
        setattr(hc, attr, new)
    x_q, s_q = hc.mix(R)
    assert (x_q - x_ref).abs().max() < 0.05 * x_ref.abs().max()
    assert (s_q - s_ref).abs().max() < 0.05 * s_ref.abs().max()


# --------------------------------------------------------------------------------------
# the shipped checkpoint: shapes and bytes
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="flash-e2m1.ftw not on this host")
def test_projection_shapes_and_bytes_match_the_shipped_checkpoint():
    with open(f"{MODEL_DIR}/config.json") as f:
        text = json.load(f)["text_config"]
    with open(f"{MODEL_DIR}/freetoken_weight.json") as f:
        tensors = {t["name"]: t for t in json.load(f)["tensors"] if t["kind"] == "weight"}
    shapes = projection_shapes(text)

    def one(pattern):
        found = [t for n, t in tensors.items() if re.search(pattern, n)]
        assert found, pattern
        assert len({tuple(t["shape"]) for t in found}) == 1, pattern
        return tuple(found[0]["shape"])

    assert shapes["attn.qkv_proj"] == one(r"self_attn\.qkv_proj")
    assert shapes["attn.o_proj"] == one(r"self_attn\.o_proj")
    assert shapes["gdn.in_proj"] == one(r"linear_attn\.in_proj")
    assert shapes["gdn.out_proj"] == one(r"linear_attn\.out_proj")
    assert shapes["shared_expert.gate_up_proj"] == one(r"shared_expert\.gate_up_proj")
    assert shapes["shared_expert.down_proj"] == one(r"shared_expert\.down_proj")
    assert shapes["hc.down_block_inject"] == one(r"hyper_connection\.input_mix_weight_down_block_inject")
    assert shapes["hc.up"] == one(r"layers\.\d+\.attn_hyper_connection\.input_mix_weight_up")
    assert shapes["hc.mixer_down"] == one(r"hyper_connection_mixer\.input_mix_weight_down\.")
    assert shapes["lm_head"] == tuple(tensors[LM_HEAD_KEY]["shape"])

    quantized = {n: t for n, t in tensors.items() if is_dense_fp8_key(n)}
    assert len(quantized) == 12 * 2 + 36 * 2 + 48 * 2 + 48 * 4 + 2
    for n, t in quantized.items():
        assert t["dtype"] == "bfloat16" and len(t["shape"]) == 2, n
    bf16 = sum(t["nbytes"] for t in quantized.values())
    assert bf16 == sum(2 * t["shape"][0] * t["shape"][1] for t in quantized.values())
    fp8_bytes = sum(t["shape"][0] * t["shape"][1] + 4 * t["shape"][0] for t in quantized.values())
    head = tensors[LM_HEAD_KEY]
    # 7.14 GB of projections (+1.27 GB lm_head) halve; the fp32 row scales are 0.4% of the fp8 bytes
    assert bf16 == 7140147200 and fp8_bytes == 3578417920
    assert head["nbytes"] == 1271398400
    # 8.41 GB of the roofline's 8.63 GB/step (the rest is router, indexer and PLE, kept BF16)
    assert bf16 + head["nbytes"] == 8411545600
    assert all(
        not is_dense_fp8_key(n) for n in tensors if "mlp.gate." in n or "indexer" in n or "ple" in n
    )


def test_prepare_for_runtime_is_a_noop_without_tuning_flag(monkeypatch):
    _, model = _build()
    monkeypatch.delenv("FREETOKEN_FP8_GEMV_TUNE", raising=False)
    model.prepare_for_runtime()  # bf16 model: nothing recorded
    model.apply_dense_weight_dtype(DenseFp8Policy(dense=True, lm_head=False))
    assert model._dense_fp8_shapes
    model.prepare_for_runtime()  # fp8 but flag unset: still no GPU work (this would raise on CPU)
