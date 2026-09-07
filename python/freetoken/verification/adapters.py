"""Capture-compatible Qwen operations retaining ordinary per-token arithmetic."""

import copy
from dataclasses import replace
from types import SimpleNamespace

def install_serial_linear(width=2):
    """Keep ordinary dense row reduction order inside a fused verification."""
    import torch
    import torch.nn.functional as functional
    from freetoken.core import get_global_ctx

    original = functional.linear

    def linear(input, weight, bias=None):
        if input.ndim == 2 and input.shape[0] == width:
            try:
                batch = get_global_ctx().batch
            except AssertionError:
                batch = None
            if getattr(batch, "mtp_fused", False):
                return torch.cat([original(input[i:i + 1], weight, bias)
                                  for i in range(width)], dim=0)
        return original(input, weight, bias)

    functional.linear = linear


def host_contexts(history, position, ids, contexts, boundary):
    """Fill token-ordered PLE contexts from known inputs, including left padding."""
    if (ids.numel() not in (2, 3, 5) or contexts.shape[0] != ids.numel()
            or history.numel() < position):
        raise ValueError("PLE staging requires matching width and complete host history")
    width = contexts.shape[1]
    contexts.fill_(boundary)
    count = min(width, position)
    if count:
        contexts[0, width - count:].copy_(history[position - count:position])
    if width:
        for step in range(1, ids.numel()):
            contexts[step, :-1].copy_(contexts[step - 1, 1:])
            contexts[step, -1].copy_(ids[step - 1])


def install_graph_support(width=2):
    """Provision PLE staging and capture-compatible metadata for fused targets.

    Retain serial width-one GDN updates. Only provision PLE's second row and make
    the two constant PLE indptrs capture-safe. Ordinary decode delegates unchanged.
    """
    import torch
    from freetoken.models.qwen4_exp import ple
    from freetoken.models.qwen4_exp.ple_uring import UringTable

    original_uring = UringTable.__init__
    original_hash = ple.NGramEmbedding.snapshot_host_hash_constants
    original_metadata = ple.build_ple_metadata
    original_conv = ple.PLELayer._short_conv
    indptrs = {}

    def indptr(device, width):
        key = (device, width)
        if key not in indptrs:
            indptrs[key] = torch.arange(0, width + 1, width, device=device, dtype=torch.int32)
        return indptrs[key]

    def uring(table, *args, **kwargs):
        kwargs["max_decode_batch_size"] = max(width, kwargs["max_decode_batch_size"])
        original_uring(table, *args, **kwargs)

    def hash_constants(embedding, max_batch_size=None):
        original_hash(embedding, max(width, max_batch_size or 0))

    def metadata(batch, args, device, context_pool=None):
        if not getattr(batch, "mtp_fused", False):
            return original_metadata(batch, args, device, context_pool)
        if context_pool is None:
            context_pool = ple._ngram_context_pool()
        slots = batch.linear_table_idx.long()
        if slots.numel() != 1 or batch.input_ids.numel() != width:
            raise ValueError("fused probe requires one request and matching token width")
        return ple.PLEMetadata(input_ids=batch.input_ids, cu_seqlens=indptr(device, width),
                               seq_lens=(width,), ngram_context=context_pool.index_select(0, slots).long(),
                               state_slots=slots, fresh_slots=None, is_decode=False, mtp_fused=True)

    def short_conv(layer, x, meta, states):
        if not meta.mtp_fused:
            return original_conv(layer, x, meta, states)
        pieces = []
        for step in range(width):
            one = ple.PLEMetadata(input_ids=meta.input_ids[step:step + 1],
                                   cu_seqlens=indptr(meta.cu_seqlens.device, 1), seq_lens=(1,),
                                   ngram_context=meta.ngram_context, state_slots=meta.state_slots,
                                   fresh_slots=None, is_decode=True)
            pieces.append(layer._decode_conv(x[step:step + 1], one, states))
        return torch.cat(pieces, dim=0)

    UringTable.__init__ = uring
    ple.NGramEmbedding.snapshot_host_hash_constants = hash_constants
    ple.build_ple_metadata = metadata
    ple.PLELayer._short_conv = short_conv


class FusedGraph:
    """A dedicated target graph with persistent token and address inputs.

    Passing a state checkpoint explicitly also makes its reads and restores follow
    the staged request slots. The caller owns these slots until acceptance or rollback finishes.
    """

    def __init__(self, engine, source, *, state_checkpoint=None):
        import torch
        from freetoken.attention.linear import build_fla_metadata

        self.engine = engine
        self.width = width = source.input_ids.numel()
        self.batch = batch = copy.copy(source)
        batch.input_ids = source.input_ids.clone()
        batch.positions = source.positions.clone()
        batch.out_loc = source.out_loc.clone()
        batch.linear_table_idx = source.linear_table_idx.clone()
        batch.active_table_idx = source.active_table_idx.clone()
        batch.fla_metadata = build_fla_metadata(batch, engine.device)
        self.state_checkpoint = state_checkpoint
        self.request_key = self._request_key(source)
        if state_checkpoint is not None:
            self.linear_state_index = batch.linear_table_idx.to(torch.int64)
            self.request_state_index = batch.active_table_idx[:1].to(torch.int64)
            state_checkpoint.bind_engine(engine, self.linear_state_index, self.request_state_index)
            state_checkpoint.state_bindings.validate_request(source.reqs[0])
        # Each row owns persistent address tensors. Replay copies new values into
        # these buffers before the captured QSA scatter plans execute.
        engine.attn_backend.prepare_metadata(batch)
        args = engine.model._config.qwen4_args
        self.ids = torch.empty(width, dtype=torch.int64, pin_memory=True)
        self.contexts = torch.empty((width, args.ngram_size - 1), dtype=torch.int64, pin_memory=True)
        self.boundary = args.ngram_boundary_token_id
        self.copy_done = torch.cuda.Event()
        self.backends = getattr(engine.model, "_ple_disk_decode", ())
        if not self.backends:
            raise RuntimeError("verification graph probe requires staged PLE backends")
        self.logits = torch.empty((width, engine.config.model_config.vocab_size),
                                  device=engine.device, dtype=torch.float32)
        self.graph = torch.cuda.CUDAGraph()
        self._prepare()
        with engine.ctx.forward_batch(batch):
            self.logits.copy_(engine.model.forward(select_last=False))
        engine.model.finish_cuda_graph_replay(record_event=True)
        engine.stream.synchronize()
        self._prepare()
        with engine.ctx.forward_batch(batch):
            with torch.cuda.graph(self.graph, stream=engine.stream):
                self.logits.copy_(engine.model.forward(select_last=False))
        engine.model.finish_cuda_graph_replay(record_event=False)
        engine.graph_runner._reset_moe_offload_cache()
        engine.stream.synchronize()

    def _prepare(self):
        import torch

        self.ids.copy_(self.batch.input_ids, non_blocking=True)
        self.copy_done.record(self.engine.stream)
        self.copy_done.synchronize()
        req = self.batch.reqs[0]
        host_contexts(req.input_ids, req.cached_len, self.ids, self.contexts, self.boundary)
        for ple, backend in self.backends:
            rows = ple.ple_embedding.host_decode_row_ids(self.contexts, self.ids)
            backend.prepare_decode(rows)
            if getattr(backend, "_decode_shape", None) != torch.Size(rows.shape):
                raise RuntimeError("verification PLE staging did not complete")

    def replay(self, batch):
        self._stage(batch)
        self._prepare()
        self.graph.replay()
        self.engine.model.finish_cuda_graph_replay(record_event=True)
        return self.logits

    @staticmethod
    def _request_key(batch):
        if len(batch.reqs) != 1 or len(batch.padded_reqs) != 1:
            raise ValueError("verification graph requires one unpadded request")
        req = batch.reqs[0]
        linear = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        return req.table_idx, linear

    def _stage(self, batch):
        request_key = self._request_key(batch)
        if not getattr(batch, "mtp_fused", False) or batch.phase != "decode":
            raise ValueError("verification graph requires a fused decode batch")
        lazy = getattr(batch.reqs[0], "lazy_kv_restore", None)
        if getattr(batch, "lazy_restore_pending", False) or (lazy is not None and not lazy.complete):
            raise ValueError("verification graph cannot replay an incomplete lazy KV restore")
        if self.state_checkpoint is None and request_key != self.request_key:
            raise RuntimeError("changing request slots requires explicit checkpoint bindings")
        if self.state_checkpoint is not None:
            self.state_checkpoint.state_bindings.validate_request(batch.reqs[0])
        copies = []
        for name in ("input_ids", "positions", "out_loc", "linear_table_idx", "active_table_idx"):
            copies.append((name, getattr(self.batch, name), getattr(batch, name)))
        if len(batch.mtp_qsa_metadata) != self.width:
            raise RuntimeError("verification graph requires metadata for every target row")
        # Restage the address/length inputs as a serving graph would. The scatter
        # plans are captured operations derived from these persistent inputs.
        for destination, source in zip(self.batch.mtp_qsa_metadata, batch.mtp_qsa_metadata):
            for name in ("block_table", "seq_lens", "ring_slots", "token_to_req", "cu_seqlens"):
                copies.append((name, getattr(destination, name), getattr(source, name)))
        # Reject incompatible inputs before partially updating any captured buffer.
        for name, destination, source in copies:
            if (source is None or destination is None or source.shape != destination.shape
                    or source.dtype != destination.dtype or source.device != destination.device):
                raise RuntimeError("verification graph input geometry changed: " + name)
        for _, destination, source in copies:
            destination.copy_(source)
        if self.state_checkpoint is not None:
            self.linear_state_index.copy_(self.batch.linear_table_idx)
            self.request_state_index.copy_(self.batch.active_table_idx[:1])
        # Host PLE staging needs the incoming history, not the capture request's.
        self.batch.reqs = list(batch.reqs)
        self.batch.padded_reqs = list(batch.padded_reqs)
        self.batch.mtp_original_cached_len = batch.mtp_original_cached_len
        self.batch.mtp_original_device_len = batch.mtp_original_device_len

    def close(self):
        self.engine.stream.synchronize()
        self.graph.reset()


def configure_fused(batch, ids, positions, locations, width):
    if ids.numel() != width or positions.numel() != width or locations.numel() != width:
        raise ValueError("fused inputs must match the configured verification width")
    req = batch.reqs[0]
    req.cached_len = batch.mtp_original_cached_len
    req.device_len = batch.mtp_original_device_len + width - 1
    batch.input_ids, batch.positions, batch.out_loc = ids, positions, locations
    batch.phase, batch.mtp_fused = "decode", True


def install_wide(width):
    """Keep width-one GDN/QSA math while the outer expert work sees all rows."""
    import torch
    from freetoken.core import get_global_ctx
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata

    original_gdn = Qwen4ExpGatedDeltaNet.forward
    original_metadata = QSASparseAttnBackend.prepare_metadata
    pin = dict(device="cpu", dtype=torch.int32, pin_memory=True)

    def gdn(layer, hidden):
        batch = get_global_ctx().batch
        if not getattr(batch, "mtp_fused", False):
            return original_gdn(layer, hidden)
        if hidden.shape[0] != width:
            raise ValueError("GDN input width differs from verification width")
        # GDN consumes hidden rows and width-one FLA metadata, not token addresses.
        # Restore the outer classification even when a primitive raises.
        batch.mtp_fused = False
        try:
            return torch.cat([original_gdn(layer, hidden[i:i + 1]) for i in range(width)], dim=0)
        finally:
            batch.mtp_fused = True

    def metadata(backend, batch):
        if not getattr(batch, "mtp_fused", False):
            return original_metadata(backend, batch)
        if batch.input_ids.numel() != width or len(batch.reqs) != 1:
            raise ValueError("QSA metadata requires one matching-width request")
        start = int(batch.mtp_original_device_len)
        batch.attn_metadata = QSASparseMetadata(
            is_decode=True, last_indices=torch.full((1,), width - 1, device=backend.device, dtype=torch.int32),
            qo_indptr_cpu=torch.tensor([0, width], **pin),
            kv_len_cpu=torch.tensor([start + width - 1], **pin))
        steps = []
        for step in range(width):
            one = QSASparseMetadata(
                is_decode=True, last_indices=torch.zeros(1, device=backend.device, dtype=torch.int32),
                qo_indptr_cpu=torch.tensor([0, 1], **pin),
                kv_len_cpu=torch.tensor([start + step], **pin))
            backend._snapshot_decode(one, batch)
            steps.append(one)
        batch.mtp_qsa_metadata = tuple(steps)

    def sparse(backend, q, k, v, index, layer_id, batch):
        if any(t.shape[0] != width for t in (q, k, v, index.k, index.q)):
            raise ValueError("QSA inputs differ from verification width")
        if len(batch.mtp_qsa_metadata) != width:
            raise ValueError("QSA metadata does not cover every target position")
        outputs = []
        for step in range(width):
            one = SimpleNamespace(
                padded_reqs=batch.padded_reqs, reqs=batch.reqs,
                lazy_restore_pending=getattr(batch, "lazy_restore_pending", False),
                positions=batch.positions[step:step + 1], out_loc=batch.out_loc[step:step + 1],
                attn_metadata=batch.mtp_qsa_metadata[step])
            one_index = replace(index, q=index.q[step:step + 1], k=index.k[step:step + 1])
            outputs.append(backend._qsa_forward_one(q[step:step + 1], k[step:step + 1],
                                                  v[step:step + 1], one_index, layer_id, one))
        return torch.cat(outputs, dim=0)

    Qwen4ExpGatedDeltaNet.forward = gdn
    QSASparseAttnBackend.prepare_metadata = metadata
    QSASparseAttnBackend._qsa_forward_mtp_k1 = sparse
