"""Qwen3.8-Flash-Next native MTP helpers.

The runtime head lives beside the target model, but greedy acceptance is kept a
small pure function so its lossless rule can be exhaustively tested without CUDA.
"""

from __future__ import annotations

import torch
from dataclasses import replace

from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
)
from freetoken.layers.moe import MoELayer
from freetoken.layers.rotary import get_rope
from freetoken.moe.fused import fused_topk

from .attention import Qwen4ExpIndexer
from .hc import GatedResidual
from .moe import Qwen4ExpMoE


from freetoken.spec_decode import MTP_DRAFT_STEPS, greedy_accept_prefix


class Qwen4ExpMTPAttention(BaseOP):
    """The MTP head's one full-attention block over its short draft chain.

    The target QSA layer uses a large paged cache. The draft head has an
    independent cache in SGLang; v1 keeps only the three-token chain locally
    and evaluates it densely. This is lossless for speculative decoding even
    when draft quality is lower, because the target verifies every proposal.
    """

    def __init__(self, config, layer_id: int) -> None:
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_attn_dim = self.num_q * self.head_dim
        self.kv_attn_dim = self.num_kv * self.head_dim
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False
        )
        self.o_proj = LinearReplicated(self.qo_attn_dim, config.hidden_size, has_bias=False)
        self.q_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary = config.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        # Loaded for checkpoint completeness. Target-aligned QSA selection is a
        # future optimization; a chain of at most three tokens is exactly dense.
        self.indexer = Qwen4ExpIndexer(config, layer_id)
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        v = v.contiguous().view(-1, self.num_kv, self.head_dim)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(
            positions, q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        q = q.view(-1, self.num_q, self.head_dim)
        k = k.view(-1, self.num_kv, self.head_dim)
        self._keys.append(k)
        self._values.append(v)
        keys = torch.cat(self._keys, dim=0).repeat_interleave(self.num_q // self.num_kv, 1)
        values = torch.cat(self._values, dim=0).repeat_interleave(
            self.num_q // self.num_kv, 1
        )
        scores = torch.einsum("thd,khd->htk", q.float(), keys.float())
        scores.mul_(self.head_dim**-0.5)
        out = torch.einsum("htk,khd->thd", scores.softmax(-1), values.float()).to(x.dtype)
        out = out.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(out)


class Qwen4ExpMTPMoE(Qwen4ExpMoE):
    """BF16 resident sibling of the target's possibly offloaded MoE block."""

    def __init__(self, config) -> None:
        resident = replace(
            config,
            moe_backend="fused",
            expert_quant="none",
            dense_quant="none",
        )
        super().__init__(resident, layer_id=None)
        assert isinstance(self.experts, MoELayer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        weights, ids = fused_topk(
            hidden_states,
            router_logits,
            self.experts.top_k,
            self.experts.renormalize,
        )
        routed = self.experts.routed_forward(hidden_states, weights, ids)
        return shared_gate_mul_add(routed, shared, gate)


class Qwen4ExpMTPDecoderLayer(BaseOP):
    def __init__(self, config) -> None:
        self.self_attn = Qwen4ExpMTPAttention(config, config.num_layers)
        self.mlp = Qwen4ExpMTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def reset(self) -> None:
        self.self_attn.reset()

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        block_output = self.self_attn.forward(block_input, positions)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTPHead(BaseOP):
    """Checkpoint-native one-layer head, reused for three greedy draft steps."""

    def __init__(self, config) -> None:
        args = config.qwen4_args
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.layers = OPList([Qwen4ExpMTPDecoderLayer(config)])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def _fuse(self, embedding: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedding = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedding))
        hidden = self.pre_fc_norm_hidden.forward(hidden)
        streams = hidden.unflatten(-1, (self.hc_count, self.hidden_size))
        streams = self.fc_hidden.forward(streams.reshape(-1, self.hidden_size)).view_as(streams)
        return (embedding.unsqueeze(-2) + streams).flatten(-2)

    def draft(
        self,
        seed_token: torch.Tensor,
        target_hidden: torch.Tensor,
        seed_position: int,
        *,
        embed_tokens,
        lm_head,
        steps: int = MTP_DRAFT_STEPS,
    ) -> torch.Tensor:
        layer = self.layers.op_list[0]
        layer.reset()
        token = seed_token.reshape(1).to(torch.int32)
        hidden = target_hidden.reshape(1, -1)
        drafts = []
        for step in range(steps):
            embedding = embed_tokens.forward(token)
            hidden = self._fuse(embedding, hidden)
            position = torch.tensor(
                [seed_position + step], dtype=torch.int32, device=token.device
            )
            hidden = layer.forward(hidden, position)
            logits = lm_head.forward(self.hyper_connection_mixer.mix(hidden)[0])
            token = torch.argmax(logits, dim=-1).to(torch.int32)
            drafts.append(token)
        return torch.cat(drafts) if drafts else token[:0]


__all__ = [
    "MTP_DRAFT_STEPS",
    "Qwen4ExpMTPHead",
    "greedy_accept_prefix",
]
