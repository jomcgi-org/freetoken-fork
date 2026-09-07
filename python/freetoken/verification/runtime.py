"""Single-owner, opt-in Qwen target verification for causal ngram proposals."""

import copy

from . import adapters, checkpoint
from .ngram import WIDTH, note_verification


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    adapters.install_serial_linear(WIDTH)
    adapters.install_graph_support(WIDTH)
    checkpoint.install()
    adapters.install_wide(WIDTH)
    _INSTALLED = True


def state_views(engine, req):
    from freetoken.spec_decode import request_state_views

    views = request_state_views(engine.linear_state_pool, engine.kv_cache, req)
    return {"conv": views["conv"], "recurrent": views["recurrent"],
            **{"slot/" + name: value for name, value in views["slot_states"].items()},
            "qsa_pending": views["qsa_pending"]}


class NgramTarget:
    def __init__(self, engine):
        self.engine = engine
        self.graph = None
        self.checkpoint = None
        self.owner = None
        self.debug_logger = None
        if engine.config.ngram_debug:
            from freetoken.utils import init_logger
            self.debug_logger = init_logger(__name__)

    def initialize(self):
        """Capture against allocated padding storage before accepting requests."""
        import torch
        from freetoken.core import Batch
        from freetoken.attention.linear import build_fla_metadata

        engine = self.engine
        if self.graph is not None or self.owner is not None:
            raise RuntimeError("ngram target was already initialized")
        if (engine.cpu_moe_executor is None or engine.linear_state_pool is None
                or engine.cpu_moe_executor.quant_format != "nvfp4"
                or sorted(engine.graph_runner.graph_map) != [1]
                or not getattr(engine.model, "_ple_disk_decode", ())):
            raise ValueError("ngram verification requires CPU MoE, staged PLE and ordinary graph size one")
        req = copy.copy(engine.dummy_req)
        req.input_ids = torch.zeros(WIDTH, dtype=torch.int32, device="cpu")
        req.cached_len, req.device_len = 0, WIDTH
        batch = Batch(reqs=[req], phase="decode")
        batch.padded_reqs = [req]
        batch.linear_table_idx = torch.tensor([req.linear_slot_idx], device=engine.device, dtype=torch.int32)
        # Scheduler._make_input_tuple supplies int64 request rows.
        batch.active_table_idx = torch.full((WIDTH,), req.table_idx, device=engine.device, dtype=torch.int64)
        batch.mtp_original_cached_len, batch.mtp_original_device_len = 0, 1
        kv = engine.kv_cache
        page_size, page = engine.config.page_size, engine.num_pages
        row = engine.page_table[req.table_idx, :page_size]
        groups = page_size // kv.index_ratio
        borrowed = [row, kv._kv_buffer.select(2, page),
                    kv._cmp_k_buffer[:, page * groups:(page + 1) * groups],
                    kv._cmp_k_buffer[:, kv.cmp_scratch_base + req.table_idx],
                    *state_views(engine, req).values()]
        engine.stream.synchronize()
        saved = [(value, value.detach().to("cpu", copy=True)) for value in borrowed]
        try:
            row.copy_(torch.arange(page * page_size, (page + 1) * page_size,
                                   dtype=row.dtype, device=row.device))
            ids = torch.zeros(WIDTH, dtype=torch.int32, device=engine.device)
            positions = torch.arange(WIDTH, dtype=torch.int32, device=engine.device)
            adapters.configure_fused(batch, ids, positions, row[:WIDTH].clone(), WIDTH)
            batch.fla_metadata = build_fla_metadata(batch, engine.device)
            engine.attn_backend.prepare_metadata(batch)
            self.checkpoint = checkpoint.SeedCheckpoint.from_engine(
                engine, req, state_views(engine, req), width=WIDTH)
            with checkpoint.capture_context(self.checkpoint):
                self.graph = adapters.FusedGraph(engine, batch, state_checkpoint=self.checkpoint)
        finally:
            engine.stream.synchronize()
            for value, prior in saved:
                value.copy_(prior)
            engine.stream.synchronize()

    def forward(self, batch):
        import torch
        from freetoken.attention.linear import build_fla_metadata
        from freetoken.spec_decode import greedy_accept_prefix

        if self.owner is not None or self.graph is None:
            raise RuntimeError("ngram target is unavailable or still owned by another batch")
        if len(batch.reqs) != 1 or len(batch.padded_reqs) != 1:
            raise ValueError("ngram verification requires one unpadded request")
        self.owner = batch
        try:
            req = batch.reqs[0]
            drafts = torch.tensor(batch.ngram_drafts, dtype=torch.int32, device=self.engine.device)
            ids = torch.cat((batch.input_ids[:1], drafts))
            adapters.configure_fused(batch, ids, batch.positions, batch.out_loc, WIDTH)
            batch.active_table_idx = batch.active_table_idx[:1].repeat(WIDTH)
            batch.fla_metadata = build_fla_metadata(batch, self.engine.device)
            self.engine.attn_backend.prepare_metadata(batch)
            logits = self.graph.replay(batch)
            targets = torch.argmax(logits, dim=-1).to(torch.int32)
            accepted, matched = greedy_accept_prefix(drafts, targets)
            # EOS and tool openers end a window before host bookkeeping. A tool
            # opener then follows ordinary decode until its exact anchor is saved.
            cuts = getattr(batch, "ngram_interrupt_ids", ())
            if cuts:
                stop = torch.zeros_like(accepted, dtype=torch.bool)
                for token in cuts:
                    stop |= accepted == token
                where = torch.nonzero(stop)
                if where.numel():
                    accepted = accepted[:int(where[0].item()) + 1]
            count = accepted.numel()
            if count < WIDTH:
                self.checkpoint.restore(count)
            req.cached_len = batch.mtp_original_cached_len + count
            req.device_len = batch.mtp_original_device_len + count
            batch.generated_tokens = count
            batch.mtp_fused = False
            # Score the draft before EOS/tool cuts; a terminal token is not a
            # failed prediction. The next probe belongs to this request only.
            note_verification(req, matched)
            if self.debug_logger is not None:
                self.debug_logger.info_rank0(f"Ngram verify: drafted={WIDTH - 1}, matched={matched}, emitted={count}")
            return accepted
        except BaseException:
            self.owner = None
            raise

    def trim(self, batch, emitted):
        """A host stop string can end inside the already verified output chunk."""
        if self.owner is not batch or not 1 <= emitted <= batch.generated_tokens:
            raise RuntimeError("ngram trim has no matching owned prefix")
        if emitted < batch.generated_tokens:
            if getattr(self, "debug_logger", None) is not None:
                self.debug_logger.info_rank0(f"Ngram host stop trim: accepted={batch.generated_tokens}, emitted={emitted}")
            self.checkpoint.restore(emitted)
            req = batch.reqs[0]
            req.cached_len = batch.mtp_original_cached_len + emitted
            req.device_len = batch.mtp_original_device_len + emitted
            batch.generated_tokens = emitted

    def release(self, batch):
        if self.owner is not batch:
            raise RuntimeError("ngram release does not own the target graph")
        self.owner = None

    def cancel(self, batch):
        if self.owner is batch:
            self.owner = None

    def close(self):
        if self.graph is not None:
            self.graph.close()
        self.graph = self.checkpoint = self.owner = None
