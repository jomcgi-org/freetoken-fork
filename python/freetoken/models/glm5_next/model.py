"""Hybrid text decoder for GLM-5.3-Flash."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.glm_moe_dsa.attention import GlmMoeDsaAttention
from freetoken.models.deepseek_v4.ops import hc_split_sinkhorn
from freetoken.utils import nvtx_annotate

from .linear import build_linear_mixer
from .mlp import Glm5NextGatedMLP
from .moe import Glm5NextSparseBlock


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm5_next_args
        self._layer_id = layer_id
        self._hidden_size = config.hidden_size
        self._hc_mult = args.hc_mult
        self._hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self._hc_eps = args.hc_eps
        self._norm_eps = config.rms_norm_eps
        self._is_linear = args.layer_types[layer_id] == "linear_attention"
        self.self_attn = (
            build_linear_mixer(config, layer_id)
            if self._is_linear
            else GlmMoeDsaAttention(config, layer_id)
        )
        self.mlp = (
            Glm5NextSparseBlock(config, layer_id)
            if args.mlp_layer_types[layer_id] == "sparse"
            else Glm5NextGatedMLP(
                config.hidden_size, config.intermediate_size, args.swiglu_limit
            )
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        mix = (2 + args.hc_mult) * args.hc_mult
        width = args.hc_mult * config.hidden_size
        self.hc_attn_fn = torch.empty(mix, width)
        self.hc_attn_base = torch.empty(mix)
        self.hc_attn_scale = torch.empty(3)
        self.hc_ffn_fn = torch.empty(mix, width)
        self.hc_ffn_base = torch.empty(mix)
        self.hc_ffn_scale = torch.empty(3)

    def _hc_pre(self, hidden, fn, base, scale):
        dtype = hidden.dtype
        flat = hidden.flatten(1).float()
        normed = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self._norm_eps
        )
        mixes = F.linear(normed, fn.float())
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            scale,
            base,
            self._hc_mult,
            self._hc_sinkhorn_iters,
            self._hc_eps,
        )
        if hidden.is_cuda:
            from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

            collapsed = hc_pre_combine(hidden, pre, dtype)
        else:
            collapsed = (pre.unsqueeze(-1) * hidden.float()).sum(1).to(dtype)
        return collapsed, post, comb

    def _hc_post(self, output, residual, post, comb):
        if output.is_cuda:
            from freetoken.kernel.triton.dsv4.hc import hc_post_combine

            return hc_post_combine(output, residual, post, comb)
        mixed = torch.matmul(comb.transpose(-1, -2), residual.float())
        return (mixed + post.unsqueeze(-1) * output.float().unsqueeze(1)).to(
            residual.dtype
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        x, post, comb = self._hc_pre(
            hidden, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale
        )
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)
        hidden = self._hc_post(x, residual, post, comb)

        residual = hidden
        x, post, comb = self._hc_pre(
            hidden, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale
        )
        x = self.post_attention_layernorm.forward(x)
        return self._hc_post(self.mlp.forward(x), residual, post, comb)


class Glm5NextModel(BaseOP):
    def __init__(self, config):
        self._hc_mult = config.glm5_next_args.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = OPList(
            [
                Glm5NextDecoderLayer(config, layer_id)
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = (
            self.embed_tokens.forward(input_ids)
            .unsqueeze(1)
            .expand(-1, self._hc_mult, -1)
            .contiguous()
        )
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        return self.norm.forward(hidden.mean(1))


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = Glm5NextModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )
        super().__init__()

    def _iter_offload_moe_layers(self):
        from freetoken.models.glm_moe import iter_glm_offload_moe_layers

        yield from iter_glm_offload_moe_layers(self)

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            if not layer._is_linear:
                layer.self_attn.prepare_for_runtime()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = [
    "Glm5NextDecoderLayer",
    "Glm5NextForCausalLM",
    "Glm5NextModel",
]
