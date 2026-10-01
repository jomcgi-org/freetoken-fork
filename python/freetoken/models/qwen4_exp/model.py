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

import os
import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger, nvtx_annotate, stage_timer

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

    def forward_attention(self, hidden: torch.Tensor, batch: Batch):
        """``forward`` up to the MLP input: (residual, MLP block input, MLP inject)."""
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return hidden, block_input, inject

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
        if getattr(self, "_capture_mtp_hidden", False):
            self._last_hc_hidden = hidden
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0]

    def forward_layer_major(
        self, batches, enter, prepare_ple, after_layer=None
    ) -> torch.Tensor:
        """Run consecutive prefill chunks of one request layer by layer.

        ``batches`` are the chunks in token order, each with its own metadata.
        ``enter(batch)`` is the context manager that makes a chunk the active batch;
        ``prepare_ple(batch)`` stages a chunk's host PLE rows; ``after_layer(i)`` runs
        once every chunk of layer ``i`` is enqueued. Every chunk passes
        layer L before any chunk reaches layer L+1, so a layer's routed experts reach
        the GPU once per group. Per chunk and per layer the operations, shapes and
        order are those of ``forward``: attention and GDN state carry from chunk to
        chunk inside each layer exactly as they do between chunk-major forwards.

        Returns the collapsed hidden state of the last chunk.
        """
        if len(self._ple) > 1:
            raise RuntimeError("layer-major prefill supports at most one PLE layer")
        hiddens = []
        for batch in batches:
            with enter(batch):
                hiddens.append(
                    self.embed_tokens.forward(batch.input_ids).repeat(1, self.hc_count)
                )
        # The request's PLE stager outlives the group: it stages later groups' chunks
        # while this group runs. A new request (or an unplanned chunk) replaces it.
        stager = getattr(self, "_ple_stager", None)
        if stager is not None and stager.index_of(batches[0]) is None:
            stager.close()
            stager = None
        if stager is None and self._ple:
            stager = _PleRequestStager.create(self._ple[0], batches)
        self._ple_stager = stager
        staged = {}

        def drop_stager():
            nonlocal stager
            if stager is not None:
                stager.close()
            stager = self._ple_stager = None

        def run_ple(layer, batch, index):
            if layer.ple is None:
                return None
            # PLE runs per chunk: a chunk's n-gram context is the previous chunk's
            # committed window.
            from .ple import build_ple_metadata

            prepare_ple(batch)
            meta = build_ple_metadata(batch, layer.ple.args, batch.input_ids.device)
            if stager is not None:
                try:
                    planned, token, local_ids = stager.take(batch)
                except Exception as exc:  # noqa: BLE001
                    if not isinstance(exc, _PleStageMismatch):
                        logger.warning_rank0(
                            "Layer-major PLE staging failed (%r); staging on the forward path",
                            exc,
                        )
                    drop_stager()
                else:
                    staged[index] = planned
                    layer.ple._pending = (meta, token)
                    layer.ple.ple_embedding.table.use_prefill_ids(token, local_ids)
                    return meta
            layer.ple.start_prefetch(batch, meta)
            return meta

        def commit_ple(meta, batch, index):
            if meta is not None:
                from .ple import commit_ngram_context

                if stager is not None and index in staged:
                    stager.release(staged.pop(index))
                commit_ngram_context(meta, getattr(batch, "fla_metadata", None))

        import os

        moe_tokens = int(os.environ.get("FREETOKEN_LAYER_MAJOR_MOE_TOKENS", "0") or 0)
        try:
            for layer_index, layer in enumerate(self.layers.op_list):
                if getattr(layer.mlp, "supports_layer_major_split", False):
                    # Pipeline within the layer: chunk c+1's attention and routing are
                    # enqueued before chunk c's experts, so waiting for c's routed rows
                    # (to stage any the prediction missed) leaves the GPU busy. Chunk c+1's
                    # attention reads only attention state that chunk c's attention wrote;
                    # each chunk runs exactly the operations of ``forward``.
                    pending = {}

                    attn_kind = "attn_linear" if layer._is_linear else "attn_full"

                    def attention(index: int) -> None:
                        batch = batches[index]
                        with enter(batch):
                            with stage_timer.span("ple"):
                                meta = run_ple(layer, batch, index)
                            with stage_timer.span(attn_kind):
                                hidden, block_input, inject = layer.forward_attention(
                                    hiddens[index], batch
                                )
                            commit_ple(meta, batch, index)
                            with stage_timer.span("moe_prepare"):
                                prepared = layer.mlp.prepare_layer_major(block_input)
                            pending[index] = (hidden, inject, prepared)

                    def experts(index: int) -> None:
                        with enter(batches[index]):
                            hidden, inject, prepared = pending.pop(index)
                            with stage_timer.span("moe_finish"):
                                out = layer.mlp.finish_layer_major(prepared)
                            with stage_timer.span("mlp_combine"):
                                hiddens[index] = layer.mlp_hyper_connection.combine(
                                    hidden, out, inject
                                )

                    if moe_tokens > 0 and hasattr(layer.mlp, "finish_layer_major_batch"):
                        # Expert batches: consecutive chunks up to moe_tokens share one
                        # routed-expert GEMM. The next batch's attention is enqueued first.
                        groups, current, size = [], [], 0
                        for index, batch in enumerate(batches):
                            tokens = int(batch.input_ids.numel())
                            if current and size + tokens > moe_tokens:
                                groups.append(current)
                                current, size = [], 0
                            current.append(index)
                            size += tokens
                        groups.append(current)

                        def experts_batch(indices) -> None:
                            with enter(batches[indices[-1]]):
                                states = [pending.pop(index) for index in indices]
                                with stage_timer.span("moe_finish"):
                                    outs = []
                                    for part in _fit_expert_batch(
                                        layer.mlp, [state[2] for state in states]
                                    ):
                                        outs.extend(layer.mlp.finish_layer_major_batch(part))
                                with stage_timer.span("mlp_combine"):
                                    for index, state, out in zip(indices, states, outs):
                                        hiddens[index] = layer.mlp_hyper_connection.combine(
                                            state[0], out, state[1]
                                        )

                        for index in groups[0]:
                            attention(index)
                        for position, indices in enumerate(groups):
                            if position + 1 < len(groups):
                                for index in groups[position + 1]:
                                    attention(index)
                            experts_batch(indices)
                    else:
                        attention(0)
                        for index in range(len(batches)):
                            if index + 1 < len(batches):
                                attention(index + 1)
                            experts(index)
                else:
                    for index, batch in enumerate(batches):
                        with enter(batch):
                            meta = run_ple(layer, batch, index)
                            hiddens[index] = layer.forward(hiddens[index], batch)
                            commit_ple(meta, batch, index)
                if after_layer is not None:
                    after_layer(layer_index)
        except BaseException:
            drop_stager()
            raise
        if stager is not None and stager.finished:
            drop_stager()
        last = hiddens[-1]
        del hiddens
        with enter(batches[-1]):
            return self.hyper_connection_mixer.mix(last)[0]


class _Ran:
    """A completed event: host-device work is already ordered."""

    def synchronize(self) -> None:
        return None


class _PleStageMismatch(RuntimeError):
    """A chunk is not one the request stager planned (its range differs)."""


class _PleRequestStager:
    """Stage a request's layer-major PLE rows on a host thread, ahead of the GPU.

    The forward path hashes a chunk's n-grams on the GPU and reads the row ids back,
    which drains every queued kernel of the group each chunk. Here a thread hashes
    each chunk from the host prompt (the same hash, ``host_prefill_row_ids``) and
    reads its unique rows into a ring of bank slots. The thread plans every
    remaining chunk of the request, so a later group's rows are read while earlier
    groups run. Chunk j's slot is restaged for chunk j + slots only after the event
    recorded behind chunk j's gather, so a gather never reads a slot being
    rewritten; the slots lie above the rows other forwards fill.
    """

    MAX_SLOTS = 16

    @classmethod
    def create(cls, ple, batches) -> "_PleRequestStager | None":
        import os

        if os.environ.get("FREETOKEN_LAYER_MAJOR_PLE_STAGE", "1") == "0":
            return None
        embedding = ple.ple_embedding
        table = getattr(embedding, "_table", None)
        prompt = getattr(batches[0], "prompt_ids", None)
        if (
            prompt is None
            or not torch.cuda.is_available()
            or not hasattr(table, "stage_prefill_slot")
            or getattr(embedding, "_host_hash_constants", None) is None
            or any(len(batch.reqs) != 1 for batch in batches)
        ):
            return None
        step = max(int(batch.reqs[0].extend_len) for batch in batches)
        heads = int(table.local_ids.shape[1])
        slots = min(cls.MAX_SLOTS, table.prefill_slots(step * heads))
        if slots < 2:
            return None
        first = batches[0].reqs[0]
        return cls(
            embedding, table, prompt, first.uid, int(first.cached_len), step, heads,
            slots, batches[0].input_ids.device,
        )

    def __init__(
        self, embedding, table, prompt, uid, start: int, step: int, heads: int,
        slots: int, device,
    ) -> None:
        import threading

        self._embedding = embedding
        self._table = table
        self._prompt = torch.as_tensor(prompt, device="cpu")
        self._uid = uid
        self._start = start
        self._step = step
        self._slot_rows = step * heads
        self._slots = slots
        self._device = torch.device(device)
        total = int(self._prompt.numel())
        self._ranges = [(b, min(b + step, total)) for b in range(start, total, step)]
        cuda = self._device.type == "cuda"
        self._out = [
            torch.empty((step, heads), dtype=torch.int64, pin_memory=cuda)
            for _ in range(slots)
        ]
        count = len(self._ranges)
        self._results: list = [None] * count
        self._ready = [threading.Event() for _ in range(count)]
        self._released = [threading.Event() for _ in range(count)]
        self._done: list = [None] * count
        self._taken = 0
        self._stop = False
        self._inference = torch.is_inference_mode_enabled()
        self._thread = threading.Thread(target=self._run, name="ple-request-stager", daemon=True)
        self._thread.start()

    def index_of(self, batch) -> int | None:
        """The planned chunk index of ``batch``, or None if it is not a planned chunk."""
        (req,) = batch.reqs
        if req.uid != self._uid:
            return None
        begin, end = int(req.cached_len), int(req.device_len)
        offset = begin - self._start
        if offset < 0 or offset % self._step:
            return None
        index = offset // self._step
        if index >= len(self._ranges) or self._ranges[index] != (begin, end):
            return None
        return index

    @property
    def finished(self) -> bool:
        """Every planned chunk's gather has been enqueued."""
        return self._released[-1].is_set() if self._released else True

    def _run(self) -> None:
        from types import SimpleNamespace

        with torch.inference_mode(self._inference):
            for index, (begin, end) in enumerate(self._ranges):
                previous = index - self._slots
                if previous >= 0:
                    self._released[previous].wait()
                    if self._stop:
                        return
                    self._done[previous].synchronize()
                if self._stop:
                    return
                try:
                    chunk = SimpleNamespace(
                        input_ids=self._prompt[:end], cached_len=begin, device_len=end
                    )
                    ids = self._embedding.host_prefill_row_ids([chunk], self._step)
                    slot = index % self._slots
                    self._results[index] = self._table.stage_prefill_slot(
                        ids, slot, self._slot_rows, self._out[slot]
                    )
                except BaseException as exc:  # noqa: BLE001
                    self._fail_from(index, exc)
                    return
                self._ready[index].set()

    def _fail_from(self, index: int, exc: BaseException) -> None:
        for later in range(index, len(self._results)):
            self._results[later] = exc
            self._ready[later].set()

    def take(self, batch):
        """The chunk's index, lookup token and bank-local ids (copied to the device)."""
        index = self.index_of(batch)
        if index is None:
            raise _PleStageMismatch(f"chunk {batch.reqs[0]} was not planned")
        # Planned chunks that ran outside a group never gather from their slot.
        for skipped in range(self._taken, index):
            if not self._released[skipped].is_set():
                self._done[skipped] = _Ran()
                self._released[skipped].set()
        self._taken = max(self._taken, index + 1)
        self._ready[index].wait()
        result = self._results[index]
        if isinstance(result, BaseException):
            raise result
        return index, object(), result.to(self._device, non_blocking=True)

    def release(self, index: int) -> None:
        """Mark chunk ``index``'s gather enqueued; its slot frees once it has run."""
        if self._device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device))
            self._done[index] = event
        else:
            self._done[index] = _Ran()
        self._released[index].set()

    def close(self) -> None:
        """Stop the thread and wait for every enqueued gather."""
        self._stop = True
        for event in self._released:
            event.set()
        self._thread.join()
        for event in self._done:
            if event is not None:
                event.synchronize()


_EXPERT_BATCH_MARGIN_BYTES = 768 << 20


def _fit_expert_batch(mlp, prepared_list):
    """Split an expert batch so its GEMM scratch fits the free GPU memory.

    The routed-expert GEMM allocates top_k x (2I + I + H) activations per token plus
    the concatenated input and output (H each), bf16. Free memory counts the CUDA
    allocator's cached-but-unused blocks, which the GEMM reuses. A batch that does
    not fit is halved until it does; a single chunk always runs as before.
    """
    if len(prepared_list) <= 1:
        return [prepared_list]
    experts = mlp.experts
    hidden = prepared_list[0][2]
    inter = getattr(experts, "intermediate_size_per_partition", None) or getattr(
        experts, "intermediate_size", 0
    )
    per_token = 2 * (experts.top_k * (3 * inter + hidden) + 2 * hidden)
    free, _total = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    budget = max(0, free - _EXPERT_BATCH_MARGIN_BYTES)

    def fits(part) -> bool:
        return sum(prepared[1] for prepared in part) * per_token <= budget

    parts, stack = [], [prepared_list]
    while stack:
        part = stack.pop(0)
        if len(part) > 1 and not fits(part):
            middle = len(part) // 2
            stack[:0] = [part[:middle], part[middle:]]
        else:
            parts.append(part)
    return parts


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

    def apply_dense_weight_dtype(self, policy):
        """``--dense-weight-dtype fp8``: swap the non-expert projections (and, per ``policy``,
        the lm_head) for per-row FP8 layers while the model is still on the meta device. The MTP
        head is built from its own BF16 layers and is not converted. Returns the report."""
        from freetoken.models.dense_fp8 import apply_dense_fp8

        report = apply_dense_fp8(self, policy)
        self._dense_fp8_shapes = report.shapes
        return report

    def prepare_for_runtime(self) -> None:
        """Engine hook, after the weights and before CUDA graph capture. With
        ``FREETOKEN_FP8_GEMV_TUNE=1`` it times the small fixed GEMV config set for every distinct
        FP8 projection shape and records the winners, so capture only ever replays fixed
        configs. Off by default (the shape heuristic is used); never fatal."""
        shapes = getattr(self, "_dense_fp8_shapes", None)
        if not shapes or os.environ.get("FREETOKEN_FP8_GEMV_TUNE") != "1":
            return
        from freetoken.kernel.triton.fp8_dense_linear import tune_gemv

        try:
            best = tune_gemv(shapes)
        except Exception as exc:  # tuning is an optimization; keep the default plans
            logger.warning(f"FP8 GEMV tuning failed ({exc!r}); using default launch configs")
            return
        for (n, k, _), plan in best.items():
            logger.info(
                f"FP8 GEMV tuned N={n} K={k}: block_n={plan.block_n} block_k={plan.block_k} "
                f"split_k={plan.split_k} warps={plan.num_warps} stages={plan.num_stages}"
            )

    def enable_mtp(self, mtp_quant: str = "bf16") -> None:
        """Build the optional head while the engine is still on the meta device."""
        from .mtp import Qwen4ExpMTPHead

        if not hasattr(self, "mtp"):
            self.mtp = Qwen4ExpMTPHead(self._config, mtp_quant)
            self.model._capture_mtp_hidden = True

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

        backend = getattr(engine_config, "ple_backend", "pinned")
        if backend == "uring":
            from .ple import process_major_faults
            from .ple_uring import UringTable, resolve_uring_source

            args = self._config.qwen4_args
            graph_sizes = getattr(engine_config, "cuda_graph_bs", None) or ()
            max_decode_batch_size = max(
                int(getattr(engine_config, "max_running_req", 1)),
                int(getattr(engine_config, "cuda_graph_max_bs", 0) or 0),
                max(graph_sizes, default=0),
            )
            max_tokens = max(
                max_decode_batch_size,
                int(
                    getattr(
                        engine_config,
                        "max_forward_len",
                        getattr(engine_config, "max_extend_tokens", 0),
                    )
                ),
            )
            required_capacity = max_tokens * args.num_ngram_heads
            source = resolve_uring_source(
                engine_config.model_path, self._config.qwen4_args
            )
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
            self._ple_waited_events = [None] * max_decode_batch_size
            reserved = 0
            staging_mib = int(
                getattr(engine_config, "ple_uring_staging_mib", 64)
            )
            for ple in ple_layers:
                streamed = UringTable(
                    source,
                    staging_mib,
                    int(getattr(engine_config, "ple_uring_queue_depth", 64)),
                    max_decode_batch_size=max_decode_batch_size,
                    rows_per_token=args.num_ngram_heads,
                    required_capacity_rows=required_capacity,
                )
                reserved += streamed.staging_nbytes
                self._ple_disk_backends.append(streamed)
                ple.ple_embedding.attach_table(streamed)
                ple.ple_embedding.snapshot_host_hash_constants(
                    max_decode_batch_size
                )
            logger.info_rank0(
                f"PLE startup: layers={len(ple_layers)}, "
                f"per_layer_budget_mib={staging_mib}, "
                f"total_resident_bytes={reserved}, "
                f"{self._ple_disk_backends[0].startup_description()}"
            )
            self._ple_disk_decode = tuple(
                zip(ple_layers, self._ple_disk_backends)
            )
            return reserved

        from .weight import load_ple_table

        table = load_ple_table(
            engine_config.model_path, self._config.qwen4_args, backend=backend,
        )
        self._ple_table = table
        if backend == "hmm":
            from .ple import HMMMappedTable, PrefillGatherTable, process_major_faults

            self._ple_hmm_backends = []
            self._ple_prefill_gather = []
            self._ple_major_fault_base = process_major_faults()
            self._ple_staging_ns = 0
            gather_on = getattr(engine_config, "ple_prefill_gather", "on") == "on"
            args = self._config.qwen4_args
            max_prefill_tokens = int(
                getattr(
                    engine_config,
                    "max_extend_tokens",
                    getattr(engine_config, "max_forward_len", 0),
                )
            )
            self._ple_prefill_max_tokens = max_prefill_tokens
            rows_per_token = int(getattr(args, "num_ngram_heads", 0))
            reserved = 0
            for ple in ple_layers:
                mapped = HMMMappedTable(table)
                if not self._ple_hmm_backends:
                    mapped.startup_probe()
                attached = mapped
                if gather_on and max_prefill_tokens > 0 and rows_per_token > 0:
                    attached = PrefillGatherTable(
                        mapped,
                        table,
                        max_prefill_tokens,
                        rows_per_token,
                    )
                    if getattr(attached, "enabled", False):
                        ple.ple_embedding.snapshot_host_hash_constants()
                        self._ple_prefill_gather.append((ple, attached))
                        reserved += int(getattr(attached, "staging_nbytes", 0))
                self._ple_hmm_backends.append(attached)
                ple.ple_embedding.attach_table(attached)
            if self._ple_prefill_gather:
                logger.info_rank0(
                    f"PLE HMM prefill gather: {max_prefill_tokens} tokens, "
                    f"{reserved / 2**20:.1f} MiB pinned across "
                    f"{len(self._ple_prefill_gather)} layer(s)"
                )
            return reserved
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
        prefill_gather = [
            table for table in backends if hasattr(table, "prefill_gather_rows")
        ]
        if prefill_gather:
            result.update({
                "ple_prefill_gather_rows": sum(
                    int(table.prefill_gather_rows) for table in prefill_gather
                ),
                "ple_prefill_gather_ms": sum(
                    float(table.prefill_gather_ms) for table in prefill_gather
                ),
            })
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
        uring = [table for table in backends if hasattr(table, "uring_stats")]
        if uring:
            stats = [table.uring_stats() for table in uring]
            requested = sum(int(item["requested_rows"]) for item in stats)
            read = sum(int(item["read_rows"]) for item in stats)
            decode_steps = max(
                (int(item["decode_fills"]) for item in stats), default=0
            )
            prefill_chunks = max(
                (int(item["prefill_fills"]) for item in stats), default=0
            )
            decode_read = sum(int(item["decode_read_rows"]) for item in stats)
            result.update({
                "ple_rows_per_step": (
                    decode_read / decode_steps if decode_steps else 0.0
                ),
                "ple_gather_ms_per_decode_step": (
                    sum(int(item["decode_gather_ns"]) for item in stats)
                    / 1_000_000.0 / decode_steps if decode_steps else 0.0
                ),
                "ple_gather_ms_per_prefill_chunk": (
                    sum(int(item["prefill_gather_ns"]) for item in stats)
                    / 1_000_000.0 / prefill_chunks if prefill_chunks else 0.0
                ),
                "ple_dedup_rate": (
                    1.0 - read / requested if requested else 0.0
                ),
            })
        if reset:
            for table in backends:
                table.reset_stats()
            self._ple_major_fault_base = now
            self._ple_staging_ns = 0
        return result

    def prepare_prefill_ple(self, batch: Batch) -> None:
        """Hash and stage one final host-side prefill chunk before model execution."""
        gather_layers = getattr(self, "_ple_prefill_gather", None)
        if not gather_layers or not getattr(batch, "is_prefill", False):
            return
        reqs = getattr(batch, "padded_reqs", None)
        if reqs is None:
            reqs = batch.reqs
        for ple, backend in gather_layers:
            try:
                row_ids = ple.ple_embedding.host_prefill_row_ids(
                    reqs,
                    int(
                        getattr(
                            backend,
                            "max_prefill_tokens",
                            getattr(self, "_ple_prefill_max_tokens", 0),
                        )
                    ),
                )
            except (MemoryError, RuntimeError) as exc:
                degrade = getattr(backend, "degrade_prefill", None)
                if degrade is not None:
                    degrade(f"host row-id allocation failed: {exc}")
                continue
            backend.prepare_prefill(row_ids)

    def cancel_prefill_ple(self) -> None:
        for _ple, backend in getattr(self, "_ple_prefill_gather", ()):
            cancel = getattr(backend, "cancel_prefill", None)
            if cancel is not None:
                cancel()

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

    @property
    def supports_layer_major_prefill(self) -> bool:
        return len(self.model.ple_layers) <= 1

    def forward_layer_major(self, batches, enter, after_layer=None) -> torch.Tensor:
        """Layer-major prefill over ``batches``; logits for the last chunk only."""
        if getattr(self.model, "_capture_mtp_hidden", False):
            raise RuntimeError("layer-major prefill does not capture MTP hidden states")
        hidden = self.model.forward_layer_major(
            batches, enter, self.prepare_prefill_ple, after_layer
        )
        with enter(batches[-1]):
            return self.lm_head.forward(hidden, select_last=True)

    def forward(self, *, select_last: bool = True) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill:
            self.prepare_prefill_ple(batch)
        else:
            self.cancel_prefill_ple()
        return self.lm_head.forward(
            self.model.forward(batch.input_ids, batch), select_last=select_last
        )


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
