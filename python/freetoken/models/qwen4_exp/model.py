"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger, nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the selected PLE backend and return bytes reserved from the pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        from .weight import load_ple_table

        backend = getattr(engine_config, "ple_backend", "pinned")
        table = load_ple_table(
            engine_config.model_path, self._config.qwen4_args, backend=backend,
        )
        self._ple_table = table
        if backend == "hmm":
            from .ple import HMMMappedTable, process_major_faults

            self._ple_hmm_backends = []
            self._ple_major_fault_base = process_major_faults()
            self._ple_staging_ns = 0
            for ple in ple_layers:
                mapped = HMMMappedTable(table)
                if not self._ple_hmm_backends:
                    mapped.startup_probe()
                self._ple_hmm_backends.append(mapped)
                ple.ple_embedding.attach_table(mapped)
            return 0
        if backend == "cached":
            from .ple import (
                CachedTable,
                load_ple_row_profile,
                ple_cache_capacity_rows,
                process_major_faults,
            )

            args = self._config.qwen4_args
            graph_sizes = getattr(engine_config, "cuda_graph_bs", None) or ()
            max_decode_batch_size = max(
                int(getattr(engine_config, "max_running_req", 1)),
                int(getattr(engine_config, "cuda_graph_max_bs", 0) or 0),
                max(graph_sizes, default=0),
            )
            max_tokens = max(
                max_decode_batch_size,
                int(engine_config.max_forward_len),
            )
            source_capacity = max_tokens * args.num_ngram_heads
            capacity = ple_cache_capacity_rows(engine_config.ple_cache_gib, table)
            decode_rows = max_decode_batch_size * args.num_ngram_heads
            if capacity < decode_rows:
                raise ValueError(
                    f"--ple-cache-gib {engine_config.ple_cache_gib} holds {capacity} rows, "
                    f"but decode graphs can require {decode_rows}; increase the cache budget"
                )
            warm_path = getattr(engine_config, "ple_cache_warm", None)
            warm_rows = (
                load_ple_row_profile(warm_path, table.num_rows) if warm_path else []
            )
            profile_out = getattr(engine_config, "ple_cache_profile_out", None)
            self._ple_disk_backends = []
            self._ple_major_fault_base = process_major_faults()
            self._ple_staging_ns = 0
            self._ple_cache_profile_out = profile_out
            pin = torch.cuda.is_available()
            self._ple_decode_contexts = torch.empty(
                (max_decode_batch_size, args.ngram_size - 1),
                dtype=torch.int64,
                pin_memory=pin,
            )
            self._ple_decode_input_ids = torch.empty(
                max_decode_batch_size, dtype=torch.int64, pin_memory=pin
            )
            self._ple_waited_events = [None] * max_decode_batch_size
            reserved = 0
            warmed = 0
            for ple in ple_layers:
                cached = CachedTable(
                    table,
                    capacity,
                    source_capacity,
                    max_decode_batch_size=max_decode_batch_size,
                    rows_per_token=args.num_ngram_heads,
                    collect_profile=bool(profile_out),
                )
                if warm_rows:
                    warmed = cached.warm(warm_rows)
                reserved += cached.cache_nbytes
                self._ple_disk_backends.append(cached)
                ple.ple_embedding.attach_table(cached)
                ple.ple_embedding.snapshot_host_hash_constants(max_decode_batch_size)
            self._ple_disk_decode = tuple(zip(ple_layers, self._ple_disk_backends))
            logger.info_rank0(
                f"PLE cache: {capacity} rows, {reserved / 2**30:.2f} GiB pinned, "
                f"{warmed} warm rows"
            )
            return reserved
        if backend == "disk":
            from .ple import DiskStagedTable, process_major_faults

            args = self._config.qwen4_args
            graph_sizes = getattr(engine_config, "cuda_graph_bs", None) or ()
            max_decode_batch_size = max(
                int(getattr(engine_config, "max_running_req", 1)),
                int(getattr(engine_config, "cuda_graph_max_bs", 0) or 0),
                max(graph_sizes, default=0),
            )
            # Prefill can need one row set per forwarded token, while decode can be
            # padded to an explicitly captured graph size larger than max_running_req.
            # Size the shared staging bank for both bounds, just like the fixed decode
            # id and hash buffers below. Dummy padding usually deduplicates, but capacity
            # must not depend on that incidental property.
            max_tokens = max(
                max_decode_batch_size,
                int(engine_config.max_forward_len),
            )
            capacity = max_tokens * args.num_ngram_heads
            self._ple_disk_backends = []
            self._ple_major_fault_base = process_major_faults()
            self._ple_staging_ns = 0
            pin = torch.cuda.is_available()
            self._ple_decode_contexts = torch.empty(
                (max_decode_batch_size, args.ngram_size - 1),
                dtype=torch.int64,
                pin_memory=pin,
            )
            self._ple_decode_input_ids = torch.empty(
                max_decode_batch_size, dtype=torch.int64, pin_memory=pin
            )
            self._ple_waited_events: list[torch.cuda.Event | None] = [
                None
            ] * max_decode_batch_size
            for ple in ple_layers:
                staged = DiskStagedTable(
                    table,
                    capacity,
                    max_decode_batch_size=max_decode_batch_size,
                    rows_per_token=args.num_ngram_heads,
                )
                self._ple_disk_backends.append(staged)
                ple.ple_embedding.attach_table(staged)
                ple.ple_embedding.snapshot_host_hash_constants(max_decode_batch_size)
            self._ple_disk_decode = tuple(zip(ple_layers, self._ple_disk_backends))
            return 0

        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table)
            )
        return table.bank.nbytes + (
            0 if table.scale_bank is None else table.scale_bank.nbytes
        )

    def ple_disk_stats(self, *, reset: bool = False) -> dict:
        """Aggregate mapped PLE prefetch and procfs major-fault counters.

        Procfs observes host-side major faults, including faults serviced through HMM,
        but does not expose GPU-side page residency directly.
        """
        backends = getattr(self, "_ple_disk_backends", None)
        if not backends:
            backends = getattr(self, "_ple_hmm_backends", None)
        if not backends:
            return {}
        from .ple import process_major_faults

        now = process_major_faults()
        base = self._ple_major_fault_base
        result = {
            "ple_prefetch_pages": sum(table.prefetch_pages for table in backends),
            "ple_major_faults": None if now is None or base is None else now - base,
            "ple_staging_us": getattr(self, "_ple_staging_ns", 0) / 1_000.0,
        }
        cached = [table for table in backends if hasattr(table, "cache_stats")]
        if cached:
            stats = [table.cache_stats() for table in cached]
            hits = sum(int(item["hits"]) for item in stats)
            misses = sum(int(item["misses"]) for item in stats)
            result.update({
                "ple_hits": hits,
                "ple_misses": misses,
                "ple_evictions": sum(int(item["evictions"]) for item in stats),
                "ple_installed_rows": sum(
                    int(item["installed_rows"]) for item in stats
                ),
                "ple_hit_rate": hits / (hits + misses) if hits + misses else 0.0,
                "ple_overflow_fallbacks": sum(
                    int(item["overflow_fallbacks"]) for item in stats
                ),
            })
            profile_out = getattr(self, "_ple_cache_profile_out", None)
            if reset and profile_out:
                from collections import Counter

                from .ple import write_ple_row_profile

                counts: Counter[int] = Counter()
                for table in cached:
                    counts.update(table.profile_counts())
                try:
                    write_ple_row_profile(profile_out, counts)
                except OSError as exc:
                    logger.warning_rank0(
                        f"could not write --ple-cache-profile-out {profile_out!r}: {exc}"
                    )
        if reset:
            for table in backends:
                table.reset_stats()
            self._ple_major_fault_base = now
            self._ple_staging_ns = 0
        return result

    def prepare_cuda_graph_replay(self, batch: Batch) -> None:
        """Stage disk PLE rows and compact ids before a decode graph replay or capture."""
        backends = getattr(self, "_ple_disk_backends", None)
        if not backends:
            return
        started = time.perf_counter_ns()
        args = self._config.qwen4_args
        context_len = args.ngram_size - 1
        batch_size = len(batch.padded_reqs)
        if batch_size > self._ple_decode_input_ids.numel():
            raise ValueError(
                f"PLE decode batch {batch_size} exceeds fixed context buffer "
                f"{self._ple_decode_input_ids.numel()}"
            )
        contexts = self._ple_decode_contexts[:batch_size]
        current_ids = self._ple_decode_input_ids[:batch_size]
        contexts.fill_(args.ngram_boundary_token_id)
        waited = self._ple_waited_events
        waited_count = 0
        for batch_index, req in enumerate(batch.padded_reqs):
            cached_len = int(req.cached_len)
            history = req.input_ids
            if history.numel() < cached_len:
                raise RuntimeError(
                    f"request {req.uid} host history ends before cached_len={cached_len}"
                )
            if cached_len < history.numel():
                current_ids[batch_index : batch_index + 1].copy_(
                    history[cached_len : cached_len + 1]
                )
            else:
                token = req.pending_token_cpu
                done = req.sample_copy_done
                if token is None or done is None:
                    # Graph capture uses the dedicated dummy request before any sample exists.
                    if req.uid != -1 or not history.numel():
                        raise RuntimeError(
                            f"decode token for request {req.uid} is not available on the host"
                        )
                    current_ids[batch_index : batch_index + 1].copy_(history[-1:])
                else:
                    already_waited = False
                    for event_index in range(waited_count):
                        prior_event = waited[event_index]
                        if done is prior_event:
                            already_waited = True
                            break
                    if not already_waited:
                        done.synchronize()
                        waited[waited_count] = done
                        waited_count += 1
                    current_ids[batch_index].copy_(token)
            prior_len = min(context_len, cached_len)
            if prior_len:
                contexts[batch_index, context_len - prior_len :].copy_(
                    history[cached_len - prior_len : cached_len]
                )

        decode_layers = getattr(self, "_ple_disk_decode", None)
        if decode_layers is None:
            decode_layers = tuple(zip(self.model.ple_layers, backends))
        assert len(decode_layers) == len(backends)
        for ple, backend in decode_layers:
            backend.prepare_decode(ple.ple_embedding.host_decode_row_ids(contexts, current_ids))
        self._ple_staging_ns += time.perf_counter_ns() - started

    def finish_cuda_graph_replay(self, *, record_event: bool) -> None:
        """Fence fixed host-buffer reuse after a submitted graph or eager warmup."""
        for backend in getattr(self, "_ple_disk_backends", ()):
            backend.finish_decode(record_event=record_event)

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
