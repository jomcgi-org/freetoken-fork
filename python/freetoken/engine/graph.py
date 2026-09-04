from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    width: int
    input_ids: torch.Tensor  # per token: [bs * width]
    out_loc: torch.Tensor  # per token: [bs * width]
    positions: torch.Tensor  # per token: [bs * width]
    logits: torch.Tensor  # per token: [bs * width, vocab_size]
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    request_table_idx: torch.Tensor  # per-token rows for session route collection
    # Per-request boundaries, with values striding by the fixed token width.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, width: int = 1
    ) -> GraphCaptureBuffer:
        if width < 1:
            raise ValueError(f"CUDA graph query width must be positive, got {width}")
        rows = bs * width
        return GraphCaptureBuffer(
            width=width,
            input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            out_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            positions=torch.zeros(rows, dtype=torch.int32, device=device),
            logits=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            # Recurrent state is keyed once per request, even when verify has multiple tokens.
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            # MoE route collection consumes one scheduler table row per query token.
            request_table_idx=torch.zeros(rows, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(
                0, (bs + 1) * width, width, dtype=torch.int32, device=device
            ),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs = batch.padded_size
        token_slice = slice(bs * self.width)
        request_slice = slice(bs)
        batch.input_ids = self.input_ids[token_slice]
        batch.out_loc = self.out_loc[token_slice]
        batch.positions = self.positions[token_slice]
        batch.linear_table_idx = self.table_idx[request_slice]
        # Route collection is per token, unlike the GDN slot map which is per
        # request, so this one follows the token slice at verify width.
        batch.active_table_idx = self.request_table_idx[token_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1],
            cache_indices=self.table_idx[request_slice],
        )

    def copy_from(self, batch: Batch) -> None:
        bs = batch.padded_size
        token_slice = slice(bs * self.width)
        request_slice = slice(bs)
        self.input_ids[token_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[token_slice] = batch.out_loc
        self.positions[token_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[request_slice] = batch.linear_table_idx
        if batch.active_table_idx is not None:
            self.request_table_idx[token_slice] = batch.active_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mtp_enabled: bool = False,
        mtp_verify_widths: List[int] | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self.model = model
        self.mtp_enabled = mtp_enabled
        self.mtp_verify_widths = (
            sorted(set(mtp_verify_widths or ())) if mtp_enabled else []
        )
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _prepare_model_replay(self, batch: Batch) -> None:
        prepare = getattr(self.model, "prepare_cuda_graph_replay", None)
        if prepare is not None:
            prepare(batch)

    def _finish_model_replay(self, *, record_event: bool) -> None:
        finish = getattr(self.model, "finish_cuda_graph_replay", None)
        if finish is not None:
            finish(record_event=record_event)

    def _prepare_mtp_model_replay(self, batch: Batch) -> bool:
        prepare = getattr(
            self.model, "prepare_mtp_verify_cuda_graph_replay", None
        )
        if prepare is None:
            self._prepare_model_replay(batch)
            return True
        return prepare(batch)

    def _mtp_model_replay_ready(self, batch: Batch) -> bool:
        ready = getattr(self.model, "mtp_verify_cuda_graph_ready", None)
        return True if ready is None else ready(batch)

    def _finish_mtp_model_replay(self, *, record_event: bool) -> None:
        finish = getattr(
            self.model, "finish_mtp_verify_cuda_graph_replay", None
        )
        if finish is None:
            self._finish_model_replay(record_event=record_event)
        else:
            finish(record_event=record_event)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_hidden_map: Dict[int, torch.Tensor] = {}
        self.mtp_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.mtp_buffer_map: Dict[int, GraphCaptureBuffer] = {}
        self.mtp_hidden_map: Dict[int, torch.Tensor] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        capture_bs_list = sorted(
            set(self.graph_bs_list + ([1] if self.mtp_verify_widths else []))
        )
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=capture_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self._prepare_model_replay(batch)
                self.buffer.logits[:bs] = model.forward()
                self._finish_model_replay(record_event=True)
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                self._prepare_model_replay(batch)
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                # Capture recorded the fixed-address gather but did not submit it.
                self._finish_model_replay(record_event=False)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
            hidden = getattr(getattr(model, "model", None), "_last_hc_hidden", None)
            if hidden is not None:
                self.graph_hidden_map[bs] = hidden

        if self.mtp_verify_widths:
            logger.info_rank0(
                "Start capturing MTP verify CUDA graphs with widths: "
                f"{self.mtp_verify_widths}"
            )
        for width in self.mtp_verify_widths:
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req], phase="decode")
            batch.padded_reqs = batch.reqs
            batch.mtp_fused = True
            batch.mtp_verify_graph_capture = True
            batch.mtp_original_cached_len = self.dummy_req.cached_len
            batch.mtp_original_device_len = self.dummy_req.cached_len + 1
            self.attn_backend.prepare_for_capture(batch)
            buffer = GraphCaptureBuffer.init(1, vocab_size, self.device, width=width)
            buffer.set_batch(batch)
            dummy_slot = (
                self.dummy_req.linear_slot_idx
                if self.dummy_req.linear_slot_idx is not None
                else self.dummy_req.table_idx
            )
            buffer.table_idx[0].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                if not self._prepare_mtp_model_replay(batch):
                    logger.warning_rank0(
                        f"Skipping MTP verify CUDA graph width {width}: "
                        "PLE staging could not be prepared"
                    )
                    self._reset_moe_offload_cache()
                    continue
                buffer.logits[:width] = model.forward(select_last=False)
                self._finish_mtp_model_replay(record_event=True)
                if not self._prepare_mtp_model_replay(batch):
                    logger.warning_rank0(
                        f"Skipping MTP verify CUDA graph width {width}: "
                        "PLE staging could not be prepared"
                    )
                    self._reset_moe_offload_cache()
                    continue
                assert self._mtp_model_replay_ready(batch), (
                    "MTP verify CUDA graph capture requires completed PLE staging"
                )
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    buffer.logits[:width] = model.forward(select_last=False)
                self._finish_mtp_model_replay(record_event=False)
                self._reset_moe_offload_cache()
            model_core = getattr(model, "model", None)
            assert getattr(model_core, "_capture_mtp_hidden", False), (
                "MTP verify CUDA graph capture requires target hidden-state capture"
            )
            hidden = getattr(model_core, "_last_hc_hidden", None)
            assert hidden is not None, "MTP verify CUDA graph did not retain target hidden state"
            self.mtp_graph_map[width] = graph
            self.mtp_buffer_map[width] = buffer
            self.mtp_hidden_map[width] = hidden
            logger.info_rank0(
                f"Captured MTP verify CUDA graph: bs=1, width={width}"
            )

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        lazy_restore_pending = getattr(batch, "lazy_restore_pending", False) or any(
            getattr(req, "lazy_kv_restore", None) is not None
            and not req.lazy_kv_restore.complete
            for req in batch.reqs
        )
        return (
            batch.is_decode
            and batch.size <= self.max_graph_bs
            and not lazy_restore_pending
        )

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        self._prepare_model_replay(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        hidden = self.graph_hidden_map.get(batch.padded_size)
        if hidden is not None:
            self.model.model._last_hc_hidden = hidden
        g.replay()
        if hidden is not None:
            assert self.model.model._last_hc_hidden is hidden
        self._finish_model_replay(record_event=True)
        return self.buffer.logits[: batch.size]

    def can_use_mtp_verify_graph(self, batch: Batch, width: int) -> bool:
        lazy_restore_pending = getattr(batch, "lazy_restore_pending", False) or any(
            getattr(req, "lazy_kv_restore", None) is not None
            and not req.lazy_kv_restore.complete
            for req in batch.reqs
        )
        return (
            self.mtp_enabled
            and width in self.mtp_graph_map
            and batch.is_decode
            and batch.size == 1
            and batch.padded_size == 1
            and getattr(batch, "mtp_fused", False)
            and batch.input_ids.numel() == width
            and not lazy_restore_pending
        )

    def replay_mtp_verify(self, batch: Batch, width: int) -> torch.Tensor | None:
        assert self.can_use_mtp_verify_graph(batch, width)
        if not self._prepare_mtp_model_replay(batch):
            return None
        assert self._mtp_model_replay_ready(batch), (
            "MTP verify CUDA graph replay requires completed PLE staging"
        )
        buffer = self.mtp_buffer_map[width]
        buffer.copy_from(batch)
        self.attn_backend.prepare_for_replay(batch)
        hidden = self.mtp_hidden_map[width]
        self.model.model._last_hc_hidden = hidden
        self.mtp_graph_map[width].replay()
        assert self.model.model._last_hc_hidden is hidden
        self._finish_mtp_model_replay(record_event=True)
        return buffer.logits[: batch.size * width]

    def pad_batch(self, batch: Batch) -> None:
        if any(
            getattr(req, "lazy_kv_restore", None) is not None
            and not req.lazy_kv_restore.complete
            for req in batch.reqs
        ):
            # Keep this batch eager even if the reader finishes between padding and submit.
            # Otherwise an unpadded size could be replayed through a graph captured at another size.
            batch.lazy_restore_pending = True
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.graph_hidden_map = {}
        self.mtp_graph_map = {}
        self.mtp_buffer_map = {}
        self.mtp_hidden_map = {}
        self.buffer = None
        gc.collect()
