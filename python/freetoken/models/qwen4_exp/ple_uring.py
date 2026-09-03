"""Strict Linux io_uring PLE backend for quantized checkpoint row layouts."""

from __future__ import annotations

import importlib
import math
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Sequence

import torch

from freetoken.utils import init_logger

from .ple import PinnedUVATable, dequantize_ple_rows
from .weight import _PleTensorPart, _ple_table_layout

logger = init_logger(__name__)

_URING_ALTERNATIVES = "--ple-backend pinned, cached, or disk"
_BOUNCE_SLOT_NBYTES = 2 * 4096


@dataclass(frozen=True)
class PleUringExtents:
    """Equal-row-count safetensors extents addressed by one global row id."""

    paths: tuple[str, ...]
    extent_file: tuple[int, ...]
    extent_base: tuple[int, ...]
    rows_per_extent: int
    row_nbytes: int
    row_stride: int

    @property
    def num_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent

    def row_offset(self, row_id: int) -> tuple[str, int]:
        """Return the exact file and byte offset for a global row id."""
        if row_id < 0 or row_id >= self.num_rows:
            raise IndexError(
                f"PLE row id must be in [0, {self.num_rows}), got {row_id}"
            )
        shard, local_row = divmod(int(row_id), self.rows_per_extent)
        path = self.paths[self.extent_file[shard]]
        return path, self.extent_base[shard] + local_row * self.row_stride


@dataclass(frozen=True)
class PleUringSource:
    """On-disk packed data, optional scale rows, and serving dequant metadata."""

    data: PleUringExtents
    scales: PleUringExtents | None
    format: str
    logical_head_dim: int
    data_dtype: torch.dtype
    stored_scale_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None
    scale_shape: tuple[int, ...]
    weight_scale: float
    shard_global_scales: tuple[float, ...] | None

    @property
    def num_rows(self) -> int:
        return self.data.num_rows

    @property
    def packed_row_nbytes(self) -> int:
        scale_bytes = 0
        if self.scale_dtype is not None:
            scale_bytes = (
                math.prod(self.scale_shape)
                * torch.empty((), dtype=self.scale_dtype).element_size()
            )
        return self.data.row_nbytes + scale_bytes

    @property
    def table_paths(self) -> tuple[str, ...]:
        paths = list(self.data.paths)
        if self.scales is not None:
            paths.extend(path for path in self.scales.paths if path not in paths)
        return tuple(paths)


def _extents_from_parts(
    parts: dict[int, _PleTensorPart], rows: int, row_nbytes: int
) -> PleUringExtents:
    if rows < 1 or row_nbytes < 1 or not parts:
        raise ValueError("PLE uring extents require positive row geometry")
    if sorted(parts) != list(range(len(parts))):
        raise ValueError(f"PLE uring shard indices are not contiguous: {sorted(parts)}")
    paths: list[str] = []
    path_ids: dict[str, int] = {}
    extent_file: list[int] = []
    extent_base: list[int] = []
    expected_bytes = rows * row_nbytes
    for shard in range(len(parts)):
        part = parts[shard]
        if part.nbytes != expected_bytes:
            raise ValueError(
                f"PLE uring shard {shard} is {part.nbytes} B, expected "
                f"{expected_bytes} B"
            )
        if part.path not in path_ids:
            path_ids[part.path] = len(paths)
            paths.append(part.path)
        extent_file.append(path_ids[part.path])
        extent_base.append(part.offset)
    return PleUringExtents(
        tuple(paths),
        tuple(extent_file),
        tuple(extent_base),
        rows,
        row_nbytes,
        row_nbytes,
    )


def resolve_uring_source(model_path: str, qwen4_args) -> PleUringSource:
    """Resolve quantized PLE payload extents without mapping or reading table rows."""
    layout = _ple_table_layout(model_path, qwen4_args)
    data_row_nbytes = (
        layout.stored_cols * torch.empty((), dtype=layout.data_dtype).element_size()
    )
    data = _extents_from_parts(layout.parts, layout.rows, data_row_nbytes)

    scales = None
    scale_shape: tuple[int, ...] = ()
    if layout.stored_scale_dtype is not None:
        scale_shape = () if layout.format == "fp8" else (layout.cols // 16,)
        stored_scale_row_nbytes = (
            math.prod(scale_shape)
            * torch.empty((), dtype=layout.stored_scale_dtype).element_size()
        )
        scales = _extents_from_parts(
            layout.scale_parts, layout.rows, stored_scale_row_nbytes
        )
    shard_global_scales = None
    if layout.global_scales is not None:
        shard_global_scales = tuple(
            float(value) for value in layout.global_scales.tolist()
        )
    return PleUringSource(
        data=data,
        scales=scales,
        format=layout.format,
        logical_head_dim=layout.cols,
        data_dtype=layout.data_dtype,
        stored_scale_dtype=layout.stored_scale_dtype,
        scale_dtype=layout.scale_dtype,
        scale_shape=scale_shape,
        weight_scale=float(layout.weight_scale),
        shard_global_scales=shard_global_scales,
    )


def _load_uring_extension() -> ModuleType:
    if sys.platform != "linux":
        raise RuntimeError(
            "--ple-backend uring requires Linux io_uring, but this platform is "
            f"{sys.platform!r}; use {_URING_ALTERNATIVES}"
        )
    try:
        return importlib.import_module("freetoken.kernel._ple_uring")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "--ple-backend uring is unavailable because the native io_uring "
            f"reader could not be loaded ({exc}); use {_URING_ALTERNATIVES}"
        ) from exc


def _make_store(extension: ModuleType, extents: PleUringExtents, queue_depth: int):
    try:
        return extension.UringRowStore(
            paths=list(extents.paths),
            extent_file=list(extents.extent_file),
            extent_base=list(extents.extent_base),
            rows_per_extent=extents.rows_per_extent,
            row_bytes=extents.row_nbytes,
            row_stride=extents.row_stride,
            queue_depth=queue_depth,
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "--ple-backend uring could not initialize io_uring "
            f"({exc}); use {_URING_ALTERNATIVES}"
        ) from exc


class UringTable:
    """PLE table streamed with strict io_uring reads into bounded pinned staging."""

    def __init__(
        self,
        source: PleUringSource,
        staging_mib: int,
        queue_depth: int,
        *,
        max_decode_batch_size: int,
        rows_per_token: int,
        required_capacity_rows: int,
        device: torch.device | None = None,
        prefetch: bool = True,
        extension: ModuleType | None = None,
    ) -> None:
        from freetoken.moe.host_banks import HostBank

        if staging_mib < 1:
            raise ValueError("PLE uring staging MiB must be positive")
        if not 1 <= queue_depth <= 4096:
            raise ValueError("PLE uring queue depth must be in [1, 4096]")
        if (
            max_decode_batch_size < 1
            or rows_per_token < 1
            or required_capacity_rows < 1
        ):
            raise ValueError(
                "PLE uring decode bounds and required capacity must be positive"
            )

        extension = extension or _load_uring_extension()
        self._data_store = _make_store(extension, source.data, queue_depth)
        self._scale_store = (
            None
            if source.scales is None
            else _make_store(extension, source.scales, queue_depth)
        )
        stores = [self._data_store]
        if self._scale_store is not None:
            stores.append(self._scale_store)

        staged_scale_bytes = source.packed_row_nbytes - source.data.row_nbytes
        raw_scale_bytes = 0
        if (
            source.scale_dtype is not None
            and source.stored_scale_dtype != source.scale_dtype
        ):
            assert source.stored_scale_dtype is not None
            raw_scale_bytes = math.prod(source.scale_shape) * torch.empty(
                (), dtype=source.stored_scale_dtype
            ).element_size()
        row_storage_bytes = (
            source.data.row_nbytes + staged_scale_bytes + raw_scale_bytes
        )
        bounce_nbytes = sum(
            int(store.queue_depth()) * _BOUNCE_SLOT_NBYTES for store in stores
        )
        local_ids_nbytes = max_decode_batch_size * rows_per_token * 4
        budget_bytes = int(staging_mib) * (1 << 20)
        global_scales_nbytes = (
            4 * len(source.shard_global_scales)
            if raw_scale_bytes and source.shard_global_scales is not None
            else 0
        )
        fixed_nbytes = bounce_nbytes + local_ids_nbytes + global_scales_nbytes
        staging_capacity = max(0, budget_bytes - fixed_nbytes) // row_storage_bytes
        capacity = min(source.num_rows, staging_capacity)
        if capacity < required_capacity_rows:
            if source.num_rows <= staging_capacity:
                raise ValueError(
                    f"PLE uring source row count {source.num_rows} bounds staging "
                    f"capacity, but this configuration can require "
                    f"{required_capacity_rows} unique rows; increasing "
                    "--ple-uring-staging-mib will not raise the row-count limit"
                )
            needed_mib = math.ceil(
                (fixed_nbytes + required_capacity_rows * row_storage_bytes)
                / (1 << 20)
            )
            raise ValueError(
                f"--ple-uring-staging-mib {staging_mib} per layer makes staging "
                f"capacity {capacity} resident rows after {fixed_nbytes} fixed bytes, "
                f"but this configuration can require {required_capacity_rows}; "
                f"use at least {needed_mib} MiB"
            )

        self.source = source
        self.num_rows = source.num_rows
        self.head_dim = source.logical_head_dim
        self.dtype = torch.bfloat16
        self.format = source.format
        self.scale = source.weight_scale
        self._capacity = capacity
        self._bounce_nbytes = bounce_nbytes
        self._rows_per_shard = source.data.rows_per_extent
        self._device = device or (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        stored_cols = (
            source.data.row_nbytes
            // torch.empty((), dtype=source.data_dtype).element_size()
        )
        self._stage_bank = HostBank((capacity, stored_cols), source.data_dtype)
        self._stage_scale_bank = None
        self._raw_scale_bank = None
        self._global_scales = None
        if source.scale_dtype is not None:
            self._stage_scale_bank = HostBank(
                (capacity, *source.scale_shape), source.scale_dtype
            )
            if source.stored_scale_dtype != source.scale_dtype:
                assert source.stored_scale_dtype is not None
                self._raw_scale_bank = HostBank(
                    (capacity, *source.scale_shape), source.stored_scale_dtype
                )
                assert source.shard_global_scales is not None
                self._global_scales = torch.tensor(
                    source.shard_global_scales, dtype=torch.float32
                )

        self._local_ids_bank = HostBank(
            (max_decode_batch_size, rows_per_token), torch.int32
        )
        self._local_ids_bank.tensor.zero_()
        probe_id = torch.zeros(1, dtype=torch.int64)
        try:
            self._data_store.read_rows(
                probe_id.data_ptr(),
                1,
                self._stage_bank.tensor.data_ptr(),
                source.data.row_nbytes,
            )
            if self._scale_store is not None:
                scale_probe = (
                    self._raw_scale_bank.tensor
                    if self._raw_scale_bank is not None
                    else self._stage_scale_bank.tensor
                )
                assert source.scales is not None
                self._scale_store.read_rows(
                    probe_id.data_ptr(),
                    1,
                    scale_probe.data_ptr(),
                    source.scales.row_nbytes,
                )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "--ple-backend uring failed its startup row-read probe "
                f"({exc}); use {_URING_ALTERNATIVES}"
            ) from exc
        self._local_ids_ptr: int | None = None
        if self._device.type == "cuda":
            from freetoken.kernel.pinned import device_ptr
            from .weight import PleTable

            self._stage_bank.pin()
            if self._stage_scale_bank is not None:
                self._stage_scale_bank.pin()
            self._local_ids_bank.pin()
            table = PleTable(
                self._stage_bank,
                torch.tensor(source.weight_scale, dtype=torch.float32),
                source.format,
                source.logical_head_dim,
                self._stage_scale_bank,
            )
            self._uva = PinnedUVATable(table, device=self._device, prefetch=prefetch)
            self._local_ids_ptr = device_ptr(self._local_ids_bank.tensor)
        else:
            self._uva = None

        self._decode_shape: torch.Size | None = None
        self._pending: tuple[torch.Tensor, torch.Tensor, str] | None = None
        self._replay_done: torch.cuda.Event | None = None
        self._poisoned = False
        self.reset_stats()

        fallbacks = list(self._data_store.direct_fallbacks())
        if self._scale_store is not None:
            fallbacks.extend(self._scale_store.direct_fallbacks())
        if fallbacks:
            logger.warning_rank0(
                "PLE uring O_DIRECT unavailable; using buffered io_uring reads: "
                + "; ".join(fallbacks)
            )

    @property
    def local_ids(self) -> torch.Tensor:
        return self._local_ids_bank.tensor

    @property
    def staging_nbytes(self) -> int:
        """Resident bank and bounce bytes charged to the host-memory governor."""
        result = (
            self._stage_bank.nbytes
            + self._local_ids_bank.nbytes
            + self._bounce_nbytes
        )
        if self._stage_scale_bank is not None:
            result += self._stage_scale_bank.nbytes
        if self._raw_scale_bank is not None:
            result += self._raw_scale_bank.nbytes
        if self._global_scales is not None:
            result += self._global_scales.nbytes
        return result

    @property
    def prefetch_pages(self) -> int:
        return 0

    def startup_description(self) -> str:
        paths = self.source.table_paths
        table_path = (
            paths[0] if len(paths) == 1 else f"{paths[0]} (+{len(paths) - 1} files)"
        )
        disk_scale_bytes = (
            0 if self.source.scales is None else self.source.scales.row_nbytes
        )
        staged_scale_bytes = self.source.packed_row_nbytes - self.source.data.row_nbytes
        row_bytes = str(self.source.data.row_nbytes + disk_scale_bytes)
        row_detail = f"data={self.source.data.row_nbytes}"
        if disk_scale_bytes:
            row_detail += f", disk_scale={disk_scale_bytes}"
        if staged_scale_bytes != disk_scale_bytes:
            row_detail += f", staged_scale={staged_scale_bytes}"
        stores = [self._data_store]
        if self._scale_store is not None:
            stores.append(self._scale_store)
        direct = all(
            store.direct_file_count() == store.file_count() for store in stores
        )
        queue_depth = min(int(store.queue_depth()) for store in stores)
        return (
            f"backend=uring, table={table_path}, format={self.format}, "
            f"row_bytes={row_bytes} ({row_detail}), "
            f"staging_bytes={self.staging_nbytes}, "
            f"queue_depth={queue_depth}, O_DIRECT={'yes' if direct else 'no'}"
        )

    def reset_stats(self) -> None:
        self._rows_requested = 0
        self._rows_read = 0
        self._decode_rows_read = 0
        self._decode_gather_ns = 0
        self._decode_fills = 0
        self._prefill_gather_ns = 0
        self._prefill_fills = 0

    def uring_stats(self) -> dict[str, int | float]:
        requested = self._rows_requested
        return {
            "requested_rows": requested,
            "read_rows": self._rows_read,
            "decode_read_rows": self._decode_rows_read,
            "decode_gather_ns": self._decode_gather_ns,
            "decode_fills": self._decode_fills,
            "prefill_gather_ns": self._prefill_gather_ns,
            "prefill_fills": self._prefill_fills,
            "dedup_rate": (1.0 - self._rows_read / requested if requested else 0.0),
        }

    def _read_unique_rows(self, unique: torch.Tensor) -> None:
        count = unique.numel()
        if self._poisoned:
            raise RuntimeError(
                "PLE uring table is poisoned after a failed fill; use "
                f"{_URING_ALTERNATIVES}"
            )
        if not count:
            return
        try:
            self._data_store.read_rows(
                unique.data_ptr(),
                count,
                self._stage_bank.tensor.data_ptr(),
                self.source.data.row_nbytes,
            )
            if self._scale_store is None:
                return
            destination = (
                self._raw_scale_bank.tensor
                if self._raw_scale_bank is not None
                else self._stage_scale_bank.tensor
            )
            assert self.source.scales is not None
            self._scale_store.read_rows(
                unique.data_ptr(),
                count,
                destination.data_ptr(),
                self.source.scales.row_nbytes,
            )
            if self._raw_scale_bank is not None:
                assert self._stage_scale_bank is not None
                serving = self._stage_scale_bank.tensor[:count]
                serving.copy_(self._raw_scale_bank.tensor[:count])
                assert self._global_scales is not None
                shard_ids = torch.div(
                    unique, self._rows_per_shard, rounding_mode="floor"
                )
                shape = (count,) + (1,) * len(self.source.scale_shape)
                serving.mul_(
                    self._global_scales.index_select(0, shard_ids).view(shape)
                )
        except (OSError, RuntimeError):
            self._poisoned = True
            raise

    def _stage_rows(
        self, row_ids: torch.Tensor | Sequence[int], *, phase: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
        flat = ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if unique.numel() > self._capacity:
            raise ValueError(
                f"PLE uring fill needs {unique.numel()} unique rows, staging holds "
                f"{self._capacity}"
            )
        if unique.numel() and (int(unique[0]) < 0 or int(unique[-1]) >= self.num_rows):
            raise IndexError(
                f"PLE row ids must be in [0, {self.num_rows}), got "
                f"[{int(unique[0])}, {int(unique[-1])}]"
            )
        started = time.perf_counter_ns()
        self._read_unique_rows(unique)
        elapsed = time.perf_counter_ns() - started
        self._rows_requested += flat.numel()
        self._rows_read += unique.numel()
        if phase == "decode":
            self._decode_rows_read += unique.numel()
            self._decode_gather_ns += elapsed
            self._decode_fills += 1
        elif phase == "prefill":
            self._prefill_gather_ns += elapsed
            self._prefill_fills += 1
        else:
            raise ValueError(f"unknown PLE uring fill phase {phase!r}")
        return unique, inverse.view_as(ids)

    def _prepare(self, row_ids: torch.Tensor, *, phase: str) -> torch.Tensor:
        _unique, inverse = self._stage_rows(row_ids.detach().cpu(), phase=phase)
        return inverse.to(row_ids.device)

    def prepare_decode(self, row_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
        if (
            ids.ndim != 2
            or ids.shape[0] > self.local_ids.shape[0]
            or ids.shape[1] != self.local_ids.shape[1]
        ):
            raise ValueError(
                f"PLE uring decode ids shape {tuple(ids.shape)} exceeds fixed buffer "
                f"{tuple(self.local_ids.shape)}"
            )
        if self._replay_done is not None:
            self._replay_done.synchronize()
        _unique, inverse = self._stage_rows(ids, phase="decode")
        self.local_ids[: ids.shape[0]].copy_(inverse)
        self._decode_shape = ids.shape

    def finish_decode(self, *, record_event: bool) -> None:
        self._decode_shape = None
        if record_event and self._device.type == "cuda":
            if self._replay_done is None:
                self._replay_done = torch.cuda.Event()
            self._replay_done.record(torch.cuda.current_stream(self._device))

    def prefetch(
        self, row_ids: torch.Tensor, *, phase: str = "prefill"
    ) -> None:
        if self._decode_shape is not None:
            return
        local_ids = self._prepare(row_ids, phase=phase)
        self._pending = (row_ids, local_ids, phase)
        if self._uva is not None:
            self._uva.prefetch(local_ids)

    def lookup(
        self, row_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._decode_shape is not None:
            if row_ids.shape != self._decode_shape:
                raise RuntimeError(
                    f"prepared uring PLE shape {tuple(self._decode_shape)} does not "
                    f"match forward shape {tuple(row_ids.shape)}"
                )
            if self._uva is not None:
                assert self._local_ids_ptr is not None
                return self._uva.lookup_from_ptr(
                    self._local_ids_ptr, self._decode_shape, out
                )
            local_ids = self.local_ids[: self._decode_shape[0]].reshape(-1).long()
            return self._lookup_cpu(local_ids, self._decode_shape, out)

        pending, self._pending = self._pending, None
        local_ids = (
            pending[1] if pending is not None and pending[0] is row_ids else None
        )
        if local_ids is None:
            # Model forwards always prefetch with the batch phase. Preserve that
            # phase if an equivalent lookup tensor misses the identity fast path.
            # A lookup with no prefetch is direct table use and follows the
            # protocol's default prefill classification.
            phase = pending[2] if pending is not None else "prefill"
            local_ids = self._prepare(row_ids, phase=phase)
        if self._uva is not None:
            return self._uva.lookup(local_ids, out)
        return self._lookup_cpu(local_ids.reshape(-1).long(), local_ids.shape, out)

    def _lookup_cpu(
        self, flat_ids: torch.Tensor, shape: torch.Size, out: torch.Tensor | None
    ) -> torch.Tensor:
        data = self._stage_bank.tensor.index_select(0, flat_ids)
        scales = None
        if self._stage_scale_bank is not None:
            scales = self._stage_scale_bank.tensor.index_select(0, flat_ids)
        rows = dequantize_ple_rows(data, scales, self.format, self.scale)
        rows = rows.view(*shape[:-1], shape[-1] * self.head_dim)
        if out is None:
            return rows
        out.copy_(rows)
        return out


__all__ = [
    "PleUringExtents",
    "PleUringSource",
    "UringTable",
    "resolve_uring_source",
]
