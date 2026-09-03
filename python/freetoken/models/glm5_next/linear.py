"""GLM-5.3 Kimi Delta Attention on FreeToken's GDN state plumbing.

The sibling Qwen GDN path supplies the varlen short-convolution, state-pool and
FLA recurrence machinery. GLM differs in four material ways: q/k/v and their
depthwise convolutions are separate checkpoint tensors, the forget projection is
low-rank, the log decay is per key dimension and safely bounded by
``lower_bound * sigmoid(exp(A_log) * x)``, and the output gate is another low-rank
projection. These differences are represented directly rather than approximated
with Qwen's scalar ``-exp(A_log) * softplus(a + dt_bias)`` gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearReplicated


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int):
        self.weight = torch.empty(dim, 1, kernel)


class _SigmoidRMSNormGated(BaseOP):
    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from freetoken.kernel.fla import rms_norm_gated

            return rms_norm_gated(
                x=x,
                weight=self.weight,
                bias=None,
                z=gate,
                eps=self.eps,
                is_rms_norm=True,
                norm_before_gate=True,
                activation="sigmoid",
            )
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float() * gate.float().sigmoid()).to(dtype)


class Glm5NextKimiDeltaAttention(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm5_next_args
        self.layer_id = layer_id
        self.num_heads = args.linear_num_heads
        self.head_dim = args.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.safe_gate_lower_bound = args.linear_lower_bound

        self.q_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.k_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.v_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.q_conv1d = _DepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)
        self.k_conv1d = _DepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)
        self.v_conv1d = _DepthwiseConv1d(self.qkv_dim, self.conv_kernel_size)

        self.f_a_proj = LinearReplicated(
            config.hidden_size, self.head_dim, has_bias=False
        )
        self.f_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.dt_bias = torch.empty(self.qkv_dim, dtype=torch.float32)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.b_proj = LinearReplicated(
            config.hidden_size, self.num_heads, has_bias=False
        )
        self.g_a_proj = LinearReplicated(
            config.hidden_size, self.head_dim, has_bias=False
        )
        self.g_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.o_norm = _SigmoidRMSNormGated(self.head_dim, config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.qkv_dim, config.hidden_size, has_bias=False)

    def _conv_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.q_conv1d.weight.squeeze(1),
                self.k_conv1d.weight.squeeze(1),
                self.v_conv1d.weight.squeeze(1),
            ),
            dim=0,
        )

    def _conv_prefill(self, conv_in, pool, fla):
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()
        out = causal_conv1d_varlen(
            x,
            self._conv_weight(),
            pool.conv_states[li],
            fla.cu_seqlens,
            fla.cache_indices,
            fla.has_initial_state,
            max_seq_len=fla.max_seq_len,
        )
        return out.transpose(0, 1)

    def _conv_decode(self, conv_in, pool, fla):
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(
            conv_in, pool.conv_states[li], self._conv_weight(), fla.cache_indices
        )

    def _gate_params(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states)).float()
        a = a.view(-1, self.num_heads, self.head_dim)
        x = a + self.dt_bias.view(1, self.num_heads, self.head_dim)
        rate = self.A_log.exp().view(1, self.num_heads, 1)
        if self.safe_gate_lower_bound is None:
            g = -rate * F.softplus(x)
        else:
            g = self.safe_gate_lower_bound * torch.sigmoid(rate * x)
        beta = self.b_proj.forward(hidden_states).float().sigmoid()
        return g, beta

    def _write_track_snapshot(self, pool, li, conv_in, h, fla) -> None:
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
        pool.conv_states[li].index_copy_(
            0, fla.track_dst, conv_win.to(pool.conv_states.dtype)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        total = hidden_states.shape[0]
        conv_in = torch.cat(
            (
                self.q_proj.forward(hidden_states),
                self.k_proj.forward(hidden_states),
                self.v_proj.forward(hidden_states),
            ),
            dim=-1,
        )
        li = pool.local_index(self.layer_id)
        if batch.is_decode:
            mixed = self._conv_decode(conv_in, pool, fla)
        else:
            mixed = self._conv_prefill(conv_in, pool, fla)
        qf, kf, vf = mixed.split([self.qkv_dim] * 3, dim=-1)
        q = qf.view(1, total, self.num_heads, self.head_dim).to(hidden_states.dtype)
        k = kf.view(1, total, self.num_heads, self.head_dim).to(hidden_states.dtype)
        v = vf.view(1, total, self.num_heads, self.head_dim).to(hidden_states.dtype)
        if batch.is_decode:
            from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

            raw_a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))
            core = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=raw_a,
                dt_bias=self.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=self.b_proj.forward(hidden_states),
                initial_state_source=pool.recurrent_states[li],
                initial_state_indices=fla.cache_indices,
                scale=self.head_dim**-0.5,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla.cu_seqlens,
                is_kda=True,
                safe_gate_lower_bound=self.safe_gate_lower_bound,
            )[0]
        else:
            from freetoken.kernel.fla import chunk_kda_with_fused_gate

            # KDA's forget gate is per channel. The Qwen chunk kernel takes one
            # gate per head, so prefill runs the vendored KDA chunk kernels, which
            # apply the same gate math the decode kernel applies in-kernel. The
            # kernel takes a gathered initial state and returns the final state;
            # the scatter back into the pool is ours.
            raw_g = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))
            beta = self.b_proj.forward(hidden_states).float().sigmoid()
            rec = pool.recurrent_states[li]
            if fla.fresh_state_indices is not None:
                rec.index_fill_(0, fla.fresh_state_indices, 0.0)
            slot_ids = fla.cache_indices.long()
            initial = rec.index_select(0, slot_ids)
            track = fla.track_dst is not None
            raw_g = raw_g.view(1, total, self.num_heads, self.head_dim)
            # The kernel validates nothing about the gate layout; this is the seam
            # that let a per-channel gate reach a per-head kernel unnoticed.
            assert raw_g.shape[-2:] == (self.num_heads, self.head_dim), raw_g.shape
            # v is a strided view of the conv output, so the kernel copies it before
            # writing its output into the copy; nothing reads mixed or v afterwards.
            result = chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=raw_g,
                beta=beta.view(1, total, self.num_heads),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                scale=self.head_dim**-0.5,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla.cu_seqlens,
                safe_gate=self.safe_gate_lower_bound is not None,
                lower_bound=(
                    self.safe_gate_lower_bound
                    if self.safe_gate_lower_bound is not None
                    else -5.0
                ),
                return_h=track,
            )
            if track:
                core, final_state, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core, final_state = result
            rec.index_copy_(0, slot_ids, final_state.to(rec.dtype))
            core = core[0]

        gate = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states)).view(
            -1, self.head_dim
        )
        out = self.o_norm.forward(core.reshape(-1, self.head_dim), gate)
        return self.o_proj.forward(out.reshape(total, self.qkv_dim))

    def forward_reference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """CPU no-cache recurrence oracle for tiny shape and formula tests."""
        bsz, seqlen, _ = hidden_states.shape
        convs = []
        for proj, conv in (
            (self.q_proj, self.q_conv1d),
            (self.k_proj, self.k_conv1d),
            (self.v_proj, self.v_conv1d),
        ):
            x = proj.forward(hidden_states).transpose(1, 2)
            convolved = F.conv1d(
                x,
                conv.weight,
                groups=self.qkv_dim,
                padding=self.conv_kernel_size - 1,
            )[..., :seqlen]
            convs.append(F.silu(convolved).transpose(1, 2))
        q, k, v = [
            x.view(bsz, seqlen, self.num_heads, self.head_dim).float() for x in convs
        ]
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        g, beta = self._gate_params(flat)
        g = g.view(bsz, seqlen, self.num_heads, self.head_dim)
        beta = beta.view(bsz, seqlen, self.num_heads)
        state = torch.zeros(
            bsz,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        rows = []
        for t in range(seqlen):
            state = state * g[:, t].exp().unsqueeze(-1)
            memory = (state * k[:, t].unsqueeze(-1)).sum(-2)
            delta = (v[:, t] - memory) * beta[:, t].unsqueeze(-1)
            state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            rows.append((state * (q[:, t] * self.head_dim**-0.5).unsqueeze(-1)).sum(-2))
        core = torch.stack(rows, dim=1).to(hidden_states.dtype)
        gate = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states)).view(
            -1, self.head_dim
        )
        out = self.o_norm.forward(core.reshape(-1, self.head_dim), gate)
        return self.o_proj.forward(out.reshape(bsz, seqlen, self.qkv_dim))


def build_linear_mixer(config, layer_id: int) -> BaseOP:
    return Glm5NextKimiDeltaAttention(config, layer_id)


__all__ = ["Glm5NextKimiDeltaAttention", "build_linear_mixer"]
