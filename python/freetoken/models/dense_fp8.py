"""``--dense-weight-dtype fp8`` / ``--fp8-lm-head``: per-row FP8 for a model's non-expert weights.

Three pieces, all keyed off one list of weight-name patterns so the module swap, the load-time
quantizer and the byte accounting can never disagree about which tensors are FP8:

* :func:`resolve_dense_fp8_policy` -- flag validation and the lm_head / MTP rule (pure).
* :func:`apply_dense_fp8` -- on the meta-device model, replaces every matching BF16 linear with a
  :class:`~freetoken.kernel.triton.fp8_dense_linear.Fp8DenseLinear` (and the lm_head with an
  ``Fp8DenseLMHead``). The model's state-dict then expects fp8 ``weight`` + fp32 ``weight_scale``.
* :func:`quantize_loaded_weights` -- wraps the checkpoint weight stream and quantizes each BF16
  tensor the model now wants as fp8, one tensor at a time, so the BF16 copy is gone before the next
  one is read and peak load memory is below the all-BF16 load it replaces.

Kept BF16: embeddings, norms, the MoE router ``mlp.gate``, ``shared_expert_gate``, the QSA indexer
projection (its logits pick the attended blocks), the PLE projections, conv1d / A_log / dt_bias,
and the whole MTP head.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

import torch

DENSE_WEIGHT_DTYPES = ("bf16", "fp8")
FP8_LM_HEAD_CHOICES = ("on", "off")

# Weight keys (checkpoint / state-dict names) of the non-expert projections that go FP8.
# Qwen3.8-Flash-Next: attention q|k|v (merged) and o_proj, GDN in_proj (qkv|z|b|a merged) and
# out_proj, shared-expert gate|up (merged) and down, and every hyper-connection projection (the
# per-layer down+inject (merged) and up, plus the top-level mixer's down and up).
DENSE_FP8_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)\.weight$",
        r"^model\.layers\.\d+\.linear_attn\.(?:in_proj|out_proj)\.weight$",
        r"^model\.layers\.\d+\.mlp\.shared_expert\.(?:gate_up_proj|down_proj)\.weight$",
        r"^model\.layers\.\d+\.(?:attn|mlp)_hyper_connection\."
        r"(?:input_mix_weight_down_block_inject|input_mix_weight_up)\.weight$",
        r"^model\.hyper_connection_mixer\.(?:input_mix_weight_down|input_mix_weight_up)\.weight$",
    )
)
LM_HEAD_KEY = "lm_head.weight"


def is_dense_fp8_key(name: str) -> bool:
    return any(p.match(name) for p in DENSE_FP8_PATTERNS)


def scale_key(weight_key: str) -> str:
    assert weight_key.endswith(".weight"), weight_key
    return weight_key[: -len("weight")] + "weight_scale"


# ======================================================================================
# Flags
# ======================================================================================
@dataclass(frozen=True)
class DenseFp8Policy:
    dense: bool  # non-expert projections are FP8
    lm_head: bool  # lm_head is FP8 (implies dense)

    @property
    def enabled(self) -> bool:
        return self.dense


def resolve_dense_fp8_policy(
    dense_weight_dtype: str, fp8_lm_head: str | None, speculative_mtp: str
) -> DenseFp8Policy:
    """Validate the flags and resolve ``--fp8-lm-head``'s default.

    The MTP draft head projects through the *same* lm_head as the target, and an FP8 lm_head
    zeroed draft acceptance in another project (sglang-rtxpro6000 issue #4). So with
    ``--speculative-mtp on`` the default is BF16, and asking for ``on`` is an error rather than a
    silent second BF16 copy that would cancel the savings. Without MTP the default follows
    ``--dense-weight-dtype fp8``.
    """
    if dense_weight_dtype not in DENSE_WEIGHT_DTYPES:
        raise ValueError(
            f"--dense-weight-dtype must be 'bf16' or 'fp8', got {dense_weight_dtype!r}"
        )
    if fp8_lm_head is not None and fp8_lm_head not in FP8_LM_HEAD_CHOICES:
        raise ValueError(f"--fp8-lm-head must be 'on' or 'off', got {fp8_lm_head!r}")
    if dense_weight_dtype == "bf16":
        if fp8_lm_head == "on":
            raise ValueError("--fp8-lm-head on requires --dense-weight-dtype fp8")
        return DenseFp8Policy(dense=False, lm_head=False)
    if fp8_lm_head == "on" and speculative_mtp == "on":
        raise ValueError(
            "--fp8-lm-head on cannot be combined with --speculative-mtp on: the MTP draft head "
            "shares the target's lm_head and an FP8 head zeroes draft acceptance. Use "
            "--fp8-lm-head off (the default under MTP)."
        )
    lm_head = (fp8_lm_head == "on") if fp8_lm_head is not None else speculative_mtp != "on"
    return DenseFp8Policy(dense=True, lm_head=lm_head)


# ======================================================================================
# Module swap (meta-device model, before load_state_dict)
# ======================================================================================
@dataclass
class DenseFp8Report:
    """What :func:`apply_dense_fp8` converted. Bytes are for the weights alone (BF16 vs
    FP8 + fp32 row scales), i.e. what the startup log reports as freed."""

    keys: list[str]
    bf16_bytes: int
    fp8_bytes: int
    lm_head: bool
    shapes: list[tuple[int, int]] = field(default_factory=list)  # distinct (N, K), for tuning

    @property
    def freed_bytes(self) -> int:
        return self.bf16_bytes - self.fp8_bytes

    def summary(self) -> str:
        gib = 2**30
        return (
            f"FP8 dense weights: {len(self.keys)} tensors"
            f"{' + lm_head' if self.lm_head else ''}, BF16 {self.bf16_bytes / gib:.2f} GiB -> "
            f"FP8 {self.fp8_bytes / gib:.2f} GiB, freed {self.freed_bytes / gib:.2f} GiB "
            f"({self.freed_bytes} bytes)"
        )


def _fp8_bytes(out_features: int, in_features: int) -> int:
    return out_features * in_features + out_features * 4  # e4m3 + fp32 row scale


def apply_dense_fp8(model, policy: DenseFp8Policy) -> DenseFp8Report:
    """Swap the matching linears of ``model`` (a ``*ForCausalLM`` with ``.model`` and ``.lm_head``)
    for FP8 layers. TP=1 only. The MTP head (``model.mtp``) is never touched."""
    from freetoken.distributed import get_tp_info
    from freetoken.kernel.triton.fp8_dense_linear import Fp8DenseLinear, Fp8DenseLMHead
    from freetoken.layers import BaseOP, OPList
    from freetoken.layers.linear import _LinearTPImpl

    if get_tp_info().size != 1:
        raise ValueError("--dense-weight-dtype fp8 currently requires --tp-size 1")
    report = DenseFp8Report(keys=[], bf16_bytes=0, fp8_bytes=0, lm_head=False)
    if not policy.dense:
        return report

    seen: set[int] = set()

    def visit(owner: BaseOP, path: str) -> None:
        if id(owner) in seen:
            return
        seen.add(id(owner))
        for name, child in list(vars(owner).items()):
            if name.startswith("_"):
                continue
            full = f"{path}.{name}" if path else name
            if isinstance(child, OPList):
                for i, op in enumerate(child.op_list):
                    visit(op, f"{full}.{i}")
            elif isinstance(child, _LinearTPImpl) and is_dense_fp8_key(full + ".weight"):
                assert child.bias is None, f"{full}: FP8 dense linears are bias-free"
                n, k = child.weight.shape
                setattr(owner, name, Fp8DenseLinear(k, n))
                report.keys.append(full + ".weight")
                report.shapes.append((n, k))
                report.bf16_bytes += n * k * 2
                report.fp8_bytes += _fp8_bytes(n, k)
            elif isinstance(child, BaseOP):
                visit(child, full)

    visit(model.model, "model")
    if policy.lm_head:
        head = model.lm_head
        assert getattr(head, "tied_embedding", None) is None, "FP8 lm_head assumes untied embeddings"
        n, k = head.weight.shape
        model.lm_head = Fp8DenseLMHead(n, k)
        report.keys.append(LM_HEAD_KEY)
        report.shapes.append((n, k))
        report.bf16_bytes += n * k * 2
        report.fp8_bytes += _fp8_bytes(n, k)
        report.lm_head = True
    return report


# ======================================================================================
# Load-time quantization (checkpoint weight stream -> fp8 + scale)
# ======================================================================================
def quantize_loaded_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    model_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pass ``weights`` through, quantizing each floating-point tensor whose model parameter is
    fp8 into ``(key, q)`` + ``(scale_key, scale)``. Streaming: one BF16 tensor is resident at a
    time, and it is dropped before the next is pulled from the checkpoint. A tensor that already
    arrives as fp8 is passed through (its scale must come from the checkpoint)."""
    from freetoken.kernel.triton.fp8_dense_linear import FP8, quantize_rowwise_fp8

    for name, tensor in weights:
        expected = model_state.get(name)
        if expected is not None and expected.dtype == FP8 and tensor.dtype != FP8:
            q, scale = quantize_rowwise_fp8(tensor.to(device=device))
            del tensor
            yield name, q
            yield scale_key(name), scale
        else:
            yield name, tensor


# ======================================================================================
# Projection shapes (microbenchmark / GPU tests / byte accounting from config.json)
# ======================================================================================
def projection_shapes(text_config: Mapping) -> dict[str, tuple[int, int]]:
    """``name -> (N, K)`` of every FP8 projection for a Qwen3.8-Flash-Next ``text_config``
    (config.json's ``text_config`` dict), derived exactly as the model constructs them."""
    hidden = text_config["hidden_size"]
    head_dim = text_config["head_dim"]
    num_q, num_kv = text_config["num_attention_heads"], text_config["num_key_value_heads"]
    key_dim = text_config["linear_num_key_heads"] * text_config["linear_key_head_dim"]
    value_dim = text_config["linear_num_value_heads"] * text_config["linear_value_head_dim"]
    num_v = text_config["linear_num_value_heads"]
    shared = text_config["shared_expert_intermediate_size"]
    hc_width = text_config["hc_count"] * hidden
    lowrank = text_config["hc_lowrank"]
    pad = (-(lowrank + text_config["hc_count"])) % 16
    return {
        "attn.qkv_proj": ((2 * num_q + 2 * num_kv) * head_dim, hidden),
        "attn.o_proj": (hidden, num_q * head_dim),
        "gdn.in_proj": (2 * key_dim + value_dim + value_dim + 2 * num_v, hidden),
        "gdn.out_proj": (hidden, value_dim),
        "shared_expert.gate_up_proj": (2 * shared, hidden),
        "shared_expert.down_proj": (hidden, shared),
        "hc.down_block_inject": (lowrank + text_config["hc_count"] + pad, hc_width),
        "hc.up": (hc_width, lowrank),
        "hc.mixer_down": (lowrank, hc_width),
        "lm_head": (text_config["vocab_size"], hidden),
    }


__all__ = [
    "DENSE_FP8_PATTERNS",
    "DENSE_WEIGHT_DTYPES",
    "DenseFp8Policy",
    "DenseFp8Report",
    "FP8_LM_HEAD_CHOICES",
    "LM_HEAD_KEY",
    "apply_dense_fp8",
    "is_dense_fp8_key",
    "projection_shapes",
    "quantize_loaded_weights",
    "resolve_dense_fp8_policy",
    "scale_key",
]
