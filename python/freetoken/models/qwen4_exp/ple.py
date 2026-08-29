"""Per-Layer Embedding (PLE) for Qwen3.8-Flash-Next: hashed n-gram features injected at layer 1.

HF reference: ``Qwen4ExpTextNGramEmbedding`` (modeling_qwen4_exp.py:1018) and
``Qwen4ExpTextPLELayer`` (:1117). Per token::

    E = table[hash(ngram)]                       # 16 heads (8 x 2-gram, 8 x 3-gram) x 160 -> 2560
    K = norm_key(key_proj(E)).view(hc, hidden)   # V = value_proj(E) [hidden]
    Q = norm_query(R).view(hc, hidden)
    u = <K_i, Q_i> / sqrt(hidden)                # per stream
    U = sigmoid(sign(u) * sqrt(max(|u|, 1e-6))) * V
    D = U + silu(conv1d(norm_conv(U)))           # depthwise, kernel 4, dilation ngram_size
    R += D                                       # before the attention hyper-connection mix

The table is the 47.7 GiB FP8 n-gram store. ``PinnedUVATable`` gathers from a fully pinned host
bank, ``HMMMappedTable`` gathers directly from read-only file mappings, and
``DiskStagedTable`` copies requested mapped rows through a small pinned bank. ``GpuResidentTable``
is the small-table oracle the host backends are diffed against.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Sequence, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated

from .config import PLE_CONV_STATE, PLE_NGRAM_STATE
from .hc import GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

    from .config import Qwen4ExpArgs


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


class _IOVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("length", ctypes.c_size_t)]


_LIBC = ctypes.CDLL(None, use_errno=True)
_PROCESS_VM_READV = getattr(_LIBC, "process_vm_readv", None)
if _PROCESS_VM_READV is not None:
    _PROCESS_VM_READV.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(_IOVec),
        ctypes.c_ulong,
        ctypes.POINTER(_IOVec),
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    _PROCESS_VM_READV.restype = ctypes.c_ssize_t
_MADVISE = getattr(_LIBC, "madvise", None)
if _MADVISE is not None:
    _MADVISE.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    _MADVISE.restype = ctypes.c_int


def process_major_faults() -> int | None:
    """Process major faults from Linux procfs, or ``None`` elsewhere.

    This includes host-side servicing attributable to HMM, but procfs does not
    expose the GPU residency of file-backed pages directly.
    """
    try:
        with open("/proc/self/stat", encoding="utf-8") as f:
            tail = f.read().rpartition(") ")[2].split()
        return int(tail[9])
    except (OSError, ValueError, IndexError):
        return None


class PLETableBackend(Protocol):
    """Row store for one PLE layer's n-gram embedding table (Qwen3.8: 40M rows x 160, FP8 + one scalar scale).

    Frozen contract. ``GpuResidentTable`` is the small-table oracle;
    ``PinnedUVATable``, ``HMMMappedTable``, and ``DiskStagedTable`` serve the real
    47.7 GiB host table. Rows are addressed by the GLOBAL hashed id, i.e. the
    per-head vocab offset is already added by ``NGramEmbedding``.

    ``lookup`` gets ``row_ids [T, num_ngram_heads]`` (int64, device) and returns
    ``[T, num_ngram_heads * head_dim]`` in ``dtype``, already dequantized (fp8 -> dtype, times the
    scalar weight_scale). ``out``, when given, is the destination and is returned as-is (CUDA-graph
    decode reuses a fixed buffer).

    ``prefetch`` may start the gather early on a side stream (the model issues it before layer 0 and
    joins it in ``lookup``); a backend with no async path makes it a no-op.
    """

    num_rows: int
    head_dim: int
    dtype: torch.dtype

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor: ...

    def prefetch(self, row_ids: torch.Tensor) -> None: ...


class GpuResidentTable:
    """PLE table held whole in GPU memory; ``index_select`` oracle for the pinned-host backend."""

    def __init__(
        self, weight: torch.Tensor, scale: float = 1.0, dtype: torch.dtype | None = None
    ) -> None:
        self.weight = weight
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape
        self.dtype = dtype if dtype is not None else (
            torch.bfloat16 if weight.dtype.itemsize < 2 else weight.dtype
        )

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = self.weight.index_select(0, row_ids.reshape(-1)).to(self.dtype)
        if self.scale != 1.0:
            rows = rows * self.scale
        rows = rows.view(*row_ids.shape[:-1], -1)
        if out is None:
            return rows
        out.copy_(rows)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None


class ZeroTable:
    """Dummy-weight stand-in: every lookup reads zeros (dummy checkpoints ship no table)."""

    def __init__(self, num_rows: int, head_dim: int, dtype: torch.dtype = torch.bfloat16) -> None:
        self.num_rows = int(num_rows)
        self.head_dim = head_dim
        self.dtype = dtype

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is not None:
            return out.zero_()
        return torch.zeros(
            (*row_ids.shape[:-1], row_ids.shape[-1] * self.head_dim),
            dtype=self.dtype,
            device=row_ids.device,
        )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None


class PinnedUVATable:
    """PLE table left in pinned host memory; rows are gathered over UVA by a Triton kernel.

    ``weight`` must be the filled and ``pin()``ed ``HostBank.tensor`` from
    ``weight.load_ple_table`` (``[num_rows, head_dim]``, fp8-e4m3 or bf16); an unregistered host
    buffer is not device-addressable and the kernel faults on it. ``scale`` is the checkpoint's
    scalar ``weight_scale``. Gathers emit bf16 into a staging buffer, one per captured decode size
    and one growable buffer for everything else.

    ``prefetch`` runs the gather on a private stream and the next ``lookup`` joins it. ``lookup``
    returns a view of that staging buffer, so the rows must be consumed before the next lookup.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: float = 1.0,
        *,
        device: torch.device | None = None,
        prefetch: bool = True,
    ) -> None:
        assert weight.device.type == "cpu" and weight.is_contiguous()
        assert weight.dtype in (torch.float8_e4m3fn, torch.bfloat16), weight.dtype
        from freetoken.kernel.pinned import device_ptr

        self.weight = weight
        self.scale = float(scale)
        self.num_rows, self.head_dim = weight.shape
        self.dtype = torch.bfloat16
        self._is_fp8 = weight.dtype == torch.float8_e4m3fn
        self._device = device or torch.device("cuda", torch.cuda.current_device())
        # WDDM maps registered host memory at a different device address; on Linux/UVA this is data_ptr
        self._table_ptr = device_ptr(weight)
        self._stream = torch.cuda.Stream(device=self._device) if prefetch else None
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        self._pending: Tuple[torch.Tensor, torch.Tensor] | None = None

    def _stage(self, rows: int) -> torch.Tensor:
        # Captured graphs keep one buffer per size for good: growing the eager one would free the
        # block a replay still writes to.
        if torch.cuda.is_current_stream_capturing():
            buf = self._graph_staging.get(rows)
            if buf is None:
                buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
                self._graph_staging[rows] = buf
            return buf
        buf = self._staging
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, self.head_dim), dtype=self.dtype, device=self._device)
            self._staging = buf
        return buf[:rows]

    def _gather(self, row_ids: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows

        return ple_gather_rows(
            self._table_ptr,
            self.num_rows,
            self.head_dim,
            row_ids.reshape(-1),
            dst,
            self.scale,
            self._is_fp8,
        )

    def _gather_from_ptr(self, row_ids_ptr: int, num_ids: int, dst: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows_from_ptr

        return ple_gather_rows_from_ptr(
            self._table_ptr,
            self.num_rows,
            self.head_dim,
            row_ids_ptr,
            num_ids,
            dst,
            self.scale,
            self._is_fp8,
        )

    def prefetch(self, row_ids: torch.Tensor) -> None:
        if self._stream is None or row_ids.numel() == 0:
            return
        dst = self._stage(row_ids.numel())
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        if not torch.cuda.is_current_stream_capturing():
            row_ids.record_stream(self._stream)
        with torch.cuda.stream(self._stream):
            self._gather(row_ids, dst)
        self._pending = (row_ids, dst)

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        pending, self._pending = self._pending, None
        if pending is not None:
            # join even on a miss: the stale prefetch owns the staging buffer about to be reused
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
        if pending is not None and pending[0] is row_ids:
            rows = pending[1]
        else:
            rows = self._gather(row_ids, self._stage(row_ids.numel()))
        rows = rows.view(*row_ids.shape[:-1], -1)
        if out is None:
            return rows
        out.copy_(rows)
        return out

    def lookup_from_ptr(
        self,
        row_ids_ptr: int,
        shape: torch.Size | tuple[int, ...],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather compact int32 ids from a fixed mapped-host buffer."""
        num_ids = math.prod(shape)
        rows = self._gather_from_ptr(row_ids_ptr, num_ids, self._stage(num_ids))
        rows = rows.view(*shape[:-1], shape[-1] * self.head_dim)
        if out is None:
            return rows
        out.copy_(rows)
        return out


def hmm_row_address(
    shard_bases: Sequence[int], row_id: int, rows_per_shard: int, row_nbytes: int
) -> int:
    """Resolve a global row id through the HMM backend's per-shard base table."""
    if rows_per_shard < 1 or row_nbytes < 1:
        raise ValueError("rows_per_shard and row_nbytes must be positive")
    num_rows = len(shard_bases) * rows_per_shard
    if row_id < 0 or row_id >= num_rows:
        raise IndexError(f"PLE row id must be in [0, {num_rows}), got {row_id}")
    shard, local_row = divmod(row_id, rows_per_shard)
    return int(shard_bases[shard]) + local_row * row_nbytes


class HMMMappedTable(PinnedUVATable):
    """Gather directly from read-only PLE shard mappings through Linux HMM.

    Safetensors payload offsets need not be page-aligned or have the same page offset.
    That prevents a generally correct back-to-back ``MAP_FIXED`` layout. A device int64
    base-pointer table therefore selects the shard for each global row id. The mapped
    shards remain unregistered and unpinned, while row ids and gather execution stay on
    the GPU and remain CUDA-graph capturable.

    The inherited prefetch overlaps the GPU gather on a private stream. There is no host
    ``MADV_WILLNEED`` path because observing live device row ids would add the round trip
    this backend is designed to avoid.
    """

    def __init__(
        self,
        table,
        *,
        device: torch.device | None = None,
        prefetch: bool = True,
    ) -> None:
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("HMM PLE requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise RuntimeError("HMM PLE requires a CUDA device")
        if not table.banks or any(
            not getattr(bank, "_disk", False) for bank in table.banks
        ):
            raise ValueError("HMM PLE requires read-only file-backed shard mappings")
        self.table = table
        self.weight = None
        self.scale = float(table.weight_scale)
        self.num_rows = int(table.num_rows)
        self.head_dim = int(table.head_dim)
        self.dtype = torch.bfloat16
        self._is_fp8 = True
        self._device = device
        self._rows_per_shard = int(table.rows_per_shard)
        self._row_nbytes = self.head_dim * torch.empty(
            (), dtype=torch.float8_e4m3fn
        ).element_size()
        self._shard_bases_host = tuple(
            int(bank.tensor.data_ptr()) for bank in table.banks
        )
        self._shard_bases = torch.tensor(
            self._shard_bases_host, dtype=torch.int64, device=self._device
        )
        self._stream = torch.cuda.Stream(device=self._device) if prefetch else None
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        self._pending: Tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def prefetch_pages(self) -> int:
        """Host page prefetch is intentionally disabled for device-only row ids."""
        return 0

    def reset_stats(self) -> None:
        return None

    def row_address(self, row_id: int) -> int:
        """Return the host virtual address used for a global row id."""
        return hmm_row_address(
            self._shard_bases_host,
            row_id,
            self._rows_per_shard,
            self._row_nbytes,
        )

    def _gather(self, row_ids: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows_sharded

        return ple_gather_rows_sharded(
            self._shard_bases,
            self._rows_per_shard,
            self.num_rows,
            self.head_dim,
            row_ids.reshape(-1),
            dst,
            self.scale,
            self._is_fp8,
        )

    def startup_probe(self) -> None:
        """Verify that the GPU can fault in and read two rows from the first shard."""
        row_nbytes = self._row_nbytes
        second = min(
            self._rows_per_shard - 1,
            max(1, (2 * mmap.PAGESIZE) // row_nbytes),
        )
        local_ids = torch.tensor(
            [[0], [second]], dtype=torch.int64, device=self._device
        )
        message = (
            "HMM PLE startup probe failed: the GPU could not read the file-backed "
            "mmap correctly. --ple-backend hmm requires the NVIDIA open GPU kernel "
            "modules; use --ple-backend disk as the fallback."
        )
        try:
            got = self.lookup(local_ids).clone()
            torch.cuda.synchronize(self._device)
            cpu_ids = local_ids.cpu().reshape(-1)
            expected = self.table.banks[0].tensor.view(torch.uint8).index_select(
                0, cpu_ids
            ).view(torch.float8_e4m3fn)
            expected = expected.to(self._device).to(torch.bfloat16)
            if self.scale != 1.0:
                expected.mul_(self.scale)
            if not torch.equal(got, expected):
                raise RuntimeError("gathered bytes did not match the CPU mapping")
        except Exception as exc:
            raise RuntimeError(message) from exc


class DiskStagedTable:
    """Read PLE rows from file mappings through a fixed pinned staging bank.

    Prefill synchronizes global row ids to the host as before. Captured decode instead receives
    host-derived ids in ``prepare_decode``. Both paths deduplicate, group by safetensors shard,
    page-prefetch, and copy once into the staging bank. Decode's fixed mapped-host compact-id
    buffer lets the captured UVA kernel preserve duplicates and input order across replays.
    """

    def __init__(
        self,
        table,
        stage_capacity_rows: int,
        *,
        device: torch.device | None = None,
        prefetch: bool = True,
        max_decode_batch_size: int | None = None,
        rows_per_token: int | None = None,
    ) -> None:
        from freetoken.moe.host_banks import HostBank

        if stage_capacity_rows < 1:
            raise ValueError("PLE disk staging capacity must be positive")
        self.table = table
        self.num_rows = int(table.num_rows)
        self.head_dim = int(table.head_dim)
        self.dtype = torch.bfloat16
        self.scale = float(table.weight_scale)
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device("cpu")
            )
        self._device = device
        self._stage_bank = HostBank(
            (int(stage_capacity_rows), self.head_dim), torch.float8_e4m3fn,
        )
        if self._device.type == "cuda":
            self._stage_bank.pin()
            self._uva = PinnedUVATable(
                self._stage_bank.tensor,
                self.scale,
                device=self._device,
                prefetch=prefetch,
            )
        else:
            self._uva = None
        self._capacity = int(stage_capacity_rows)
        self._prefetch_pages = 0
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._local_ids_bank = None
        self._local_ids_ptr: int | None = None
        self._decode_shape: torch.Size | None = None
        self._replay_done: torch.cuda.Event | None = None
        self._rows_per_shard = int(table.rows_per_shard)
        self._row_nbytes = self.head_dim * torch.empty(
            (), dtype=torch.float8_e4m3fn
        ).element_size()
        self._shard_bases = torch.tensor(
            [bank.tensor.data_ptr() for bank in table.banks], dtype=torch.int64
        )
        self._has_disk_banks = any(getattr(bank, "_disk", False) for bank in table.banks)
        self._iov_max = max(1, int(os.sysconf("SC_IOV_MAX")))
        self._process_vm_readv = _PROCESS_VM_READV
        if max_decode_batch_size is not None or rows_per_token is not None:
            if not max_decode_batch_size or not rows_per_token:
                raise ValueError(
                    "max_decode_batch_size and rows_per_token must both be positive"
                )
            self._local_ids_bank = HostBank(
                (int(max_decode_batch_size), int(rows_per_token)), torch.int32,
            )
            scratch_rows = int(max_decode_batch_size) * int(rows_per_token)
            self._sorted_ids = torch.empty(scratch_rows, dtype=torch.int64)
            self._sort_order = torch.empty(scratch_rows, dtype=torch.int64)
            self._unique_flags = torch.empty(scratch_rows, dtype=torch.bool)
            self._sorted_local_ids = torch.empty(scratch_rows, dtype=torch.int64)
            self._inverse_ids = torch.empty(scratch_rows, dtype=torch.int64)
            self._unique_ids = torch.empty(scratch_rows, dtype=torch.int64)
            self._shard_ids = torch.empty(scratch_rows, dtype=torch.int64)
            self._local_rows = torch.empty(scratch_rows, dtype=torch.int64)
            # Linux copies all discontiguous mmap rows into the contiguous staging bank with
            # one process_vm_readv syscall per IOV_MAX rows. The interleaved int64 layout is
            # exactly struct iovec on the supported 64-bit serving platforms.
            self._remote_iov = torch.empty((scratch_rows, 2), dtype=torch.int64)
            self._remote_iov[:, 1].fill_(self._row_nbytes)
            self._page_candidates = torch.empty(scratch_rows * 2, dtype=torch.int64)
            self._page_sorted = torch.empty(scratch_rows * 2, dtype=torch.int64)
            self._page_order = torch.empty(scratch_rows * 2, dtype=torch.int64)
            self._page_flags = torch.empty(scratch_rows * 2, dtype=torch.bool)
            self._page_local_ids = torch.empty(scratch_rows * 2, dtype=torch.int64)
            self._unique_pages = torch.empty(scratch_rows * 2, dtype=torch.int64)
            if self._device.type == "cuda":
                from freetoken.kernel.pinned import device_ptr

                self._local_ids_bank.pin()
                self._local_ids_ptr = device_ptr(self._local_ids_bank.tensor)

    @property
    def prefetch_pages(self) -> int:
        return self._prefetch_pages

    def reset_stats(self) -> None:
        self._prefetch_pages = 0

    @property
    def local_ids(self) -> torch.Tensor | None:
        """Fixed compact-id buffer, exposed for CPU staging tests and diagnostics."""
        return None if self._local_ids_bank is None else self._local_ids_bank.tensor

    def _stage_rows_reference(
        self, row_ids: torch.Tensor | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slow staging oracle retained for prefill and decode micro-benchmark parity."""
        ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
        shape = ids.shape
        flat = ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if unique.numel() > self._capacity:
            raise ValueError(
                f"PLE lookup needs {unique.numel()} unique rows, staging holds "
                f"{self._capacity}"
            )
        if unique.numel() and (int(unique[0]) < 0 or int(unique[-1]) >= self.num_rows):
            raise IndexError(
                f"PLE row ids must be in [0, {self.num_rows}), got "
                f"[{int(unique[0])}, {int(unique[-1])}]"
            )

        staged = self._stage_bank.tensor[: unique.numel()]
        staged_bytes = staged.view(torch.uint8)
        rows_per_shard = int(self.table.rows_per_shard)
        pages = 0
        shard_ids = torch.div(unique, rows_per_shard, rounding_mode="floor")
        for shard in torch.unique(shard_ids).tolist():
            bank = self.table.banks[shard]
            lo = shard * rows_per_shard
            positions = (shard_ids == shard).nonzero().reshape(-1)
            local = unique.index_select(0, positions) - lo
            pages += bank.prefetch_rows(local.tolist())
            source = bank.tensor.view(torch.uint8).index_select(0, local)
            staged_bytes.index_copy_(0, positions, source)
        self._prefetch_pages += pages
        return staged, inverse.view(shape)

    def _stage_rows(
        self, row_ids: torch.Tensor | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility entry point for the prefill slow path."""
        return self._stage_rows_reference(row_ids)

    def _prefetch_decode_pages(self, addresses: torch.Tensor) -> int:
        """Deduplicate row pages with fixed tensor buffers, then issue WILLNEED."""
        if not self._has_disk_banks or not addresses.numel() or _MADVISE is None:
            return 0
        count = addresses.numel()
        page_size = mmap.PAGESIZE
        candidates = self._page_candidates[: count * 2]
        candidates[:count].copy_(addresses)
        candidates[:count].floor_divide_(page_size).mul_(page_size)
        candidates[count:].copy_(addresses).add_(self._row_nbytes - 1)
        candidates[count:].floor_divide_(page_size).mul_(page_size)
        torch.sort(
            candidates,
            out=(self._page_sorted[: count * 2], self._page_order[: count * 2]),
        )
        ordered = self._page_sorted[: count * 2]
        flags = self._page_flags[: count * 2]
        flags[0] = True
        if ordered.numel() > 1:
            torch.ne(ordered[1:], ordered[:-1], out=flags[1:])
        torch.cumsum(
            flags,
            dim=0,
            dtype=torch.int64,
            out=self._page_local_ids[: count * 2],
        )
        page_ids = self._page_local_ids[: count * 2].sub_(1)
        page_count = int(page_ids[-1]) + 1
        pages = self._unique_pages[:page_count]
        pages.scatter_(0, page_ids, ordered)

        # The mapped shards have unrelated virtual addresses, so madvise needs one call per
        # contiguous virtual range. The page set and range boundaries are already deduplicated.
        start = previous = int(pages[0])
        for index in range(1, page_count + 1):
            current = int(pages[index]) if index < page_count else -1
            if current == previous + page_size:
                previous = current
                continue
            length = previous - start + page_size
            if _MADVISE(ctypes.c_void_p(start), length, mmap.MADV_WILLNEED):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            start = previous = current
        return page_count

    def _copy_decode_rows(self, addresses: torch.Tensor, row_count: int) -> None:
        """Copy discontiguous source rows into the fixed contiguous staging bank."""
        dst = self._stage_bank.tensor.data_ptr()
        if self._process_vm_readv is None:
            for index in range(row_count):
                ctypes.memmove(
                    dst + index * self._row_nbytes,
                    int(addresses[index]),
                    self._row_nbytes,
                )
            return

        copied_rows = 0
        while copied_rows < row_count:
            chunk_rows = min(self._iov_max, row_count - copied_rows)
            chunk_bytes = chunk_rows * self._row_nbytes
            local = _IOVec(
                ctypes.c_void_p(dst + copied_rows * self._row_nbytes), chunk_bytes
            )
            remote = ctypes.cast(
                self._remote_iov.data_ptr()
                + copied_rows * ctypes.sizeof(_IOVec),
                ctypes.POINTER(_IOVec),
            )
            copied = self._process_vm_readv(
                os.getpid(), ctypes.byref(local), 1, remote, chunk_rows, 0
            )
            if copied != chunk_bytes:
                # Some container seccomp profiles block process_vm_readv even for self. Fall
                # back once and remember it, rather than making serving depend on that policy.
                self._process_vm_readv = None
                for index in range(row_count):
                    ctypes.memmove(
                        dst + index * self._row_nbytes,
                        int(addresses[index]),
                        self._row_nbytes,
                    )
                return
            copied_rows += chunk_rows

    def _stage_decode_rows(self, ids: torch.Tensor) -> int:
        """Allocation-free sort/dedup, page prefetch, and batched row copy for decode."""
        flat = ids.reshape(-1)
        count = flat.numel()
        if count > self._sorted_ids.numel():
            raise ValueError(
                f"PLE decode needs {count} rows, fixed buffers hold "
                f"{self._sorted_ids.numel()}"
            )
        if not count:
            return 0

        sorted_ids = self._sorted_ids[:count]
        sort_order = self._sort_order[:count]
        torch.sort(flat, out=(sorted_ids, sort_order))
        if int(sorted_ids[0]) < 0 or int(sorted_ids[-1]) >= self.num_rows:
            raise IndexError(
                f"PLE row ids must be in [0, {self.num_rows}), got "
                f"[{int(sorted_ids[0])}, {int(sorted_ids[-1])}]"
            )
        flags = self._unique_flags[:count]
        flags[0] = True
        if count > 1:
            torch.ne(sorted_ids[1:], sorted_ids[:-1], out=flags[1:])
        torch.cumsum(
            flags,
            dim=0,
            dtype=torch.int64,
            out=self._sorted_local_ids[:count],
        )
        sorted_local_ids = self._sorted_local_ids[:count].sub_(1)
        unique_count = int(sorted_local_ids[-1]) + 1
        if unique_count > self._capacity:
            raise ValueError(
                f"PLE lookup needs {unique_count} unique rows, staging holds "
                f"{self._capacity}"
            )
        unique = self._unique_ids[:unique_count]
        unique.scatter_(0, sorted_local_ids, sorted_ids)
        inverse = self._inverse_ids[:count]
        inverse.scatter_(0, sort_order, sorted_local_ids)
        local_ids = self.local_ids
        assert local_ids is not None
        local_ids[: ids.shape[0]].copy_(inverse.view_as(ids))

        shard_ids = self._shard_ids[:unique_count]
        local_rows = self._local_rows[:unique_count]
        torch.div(
            unique,
            self._rows_per_shard,
            rounding_mode="floor",
            out=shard_ids,
        )
        torch.remainder(unique, self._rows_per_shard, out=local_rows)
        addresses = self._remote_iov[:unique_count, 0]
        torch.index_select(self._shard_bases, 0, shard_ids, out=addresses)
        addresses.add_(local_rows, alpha=self._row_nbytes)
        self._prefetch_pages += self._prefetch_decode_pages(addresses)
        self._copy_decode_rows(addresses, unique_count)
        return unique_count

    def _prepare(self, row_ids: torch.Tensor) -> torch.Tensor:
        _staged, inverse = self._stage_rows(row_ids.detach().cpu())
        return inverse.to(row_ids.device)

    def prepare_decode(self, row_ids: torch.Tensor) -> None:
        """Run all disk and dedup work before a decode graph replay.

        ``row_ids`` is the host-derived global-id matrix for the padded graph batch. The staging
        bank and local-id bank are overwritten in place, then the captured gather reads both at
        their fixed mapped addresses.
        """
        if self._local_ids_bank is None:
            raise RuntimeError("disk PLE decode buffers were not provisioned")
        ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
        local_ids = self.local_ids
        assert local_ids is not None
        if (
            ids.ndim != 2
            or ids.shape[0] > local_ids.shape[0]
            or ids.shape[1] != local_ids.shape[1]
        ):
            raise ValueError(
                f"PLE decode ids shape {tuple(ids.shape)} exceeds fixed buffer "
                f"{tuple(local_ids.shape)}"
            )
        # Host writes must not race a previous replay that is still reading these mapped buffers.
        if self._replay_done is not None:
            self._replay_done.synchronize()
        self._stage_decode_rows(ids)
        self._decode_shape = ids.shape

    def finish_decode(self, *, record_event: bool) -> None:
        """Release Python-side decode mode and optionally fence a submitted replay."""
        self._decode_shape = None
        if record_event and self._device.type == "cuda":
            if self._replay_done is None:
                self._replay_done = torch.cuda.Event()
            self._replay_done.record(torch.cuda.current_stream(self._device))

    def prefetch(self, row_ids: torch.Tensor) -> None:
        if self._decode_shape is not None:
            # The graph's lookup launches the fixed-address gather. All host work is complete.
            return
        local_ids = self._prepare(row_ids)
        self._pending = (row_ids, local_ids)
        if self._uva is not None:
            self._uva.prefetch(local_ids)

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if self._decode_shape is not None:
            if row_ids.shape != self._decode_shape:
                raise RuntimeError(
                    f"prepared disk PLE shape {tuple(self._decode_shape)} does not match "
                    f"forward shape {tuple(row_ids.shape)}"
                )
            if self._uva is not None:
                assert self._local_ids_ptr is not None
                return self._uva.lookup_from_ptr(self._local_ids_ptr, self._decode_shape, out)
            local_ids = self.local_ids[: self._decode_shape[0]]
            staged = self._stage_bank.tensor
            rows = staged.view(torch.uint8).index_select(
                0, local_ids.reshape(-1).long()
            ).view(torch.float8_e4m3fn).to(torch.bfloat16)
            if self.scale != 1.0:
                rows.mul_(self.scale)
            rows = rows.view(*self._decode_shape[:-1], -1)
            if out is None:
                return rows
            out.copy_(rows)
            return out
        pending, self._pending = self._pending, None
        local_ids = pending[1] if pending is not None and pending[0] is row_ids else None
        if local_ids is None:
            local_ids = self._prepare(row_ids)
        if self._uva is not None:
            return self._uva.lookup(local_ids, out)

        staged = self._stage_bank.tensor
        rows = staged.view(torch.uint8).index_select(
            0, local_ids.reshape(-1)
        ).view(torch.float8_e4m3fn).to(torch.bfloat16)
        if self.scale != 1.0:
            rows.mul_(self.scale)
        rows = rows.view(*local_ids.shape[:-1], -1)
        if out is None:
            return rows
        out.copy_(rows)
        return out


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _derive_decode_row_ids_host_reference(
    contexts: Sequence[Sequence[int]],
    input_ids: Sequence[int],
    *,
    layer_multipliers: Sequence[int],
    vocab_sizes: Sequence[int],
    offsets: Sequence[int],
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Original scalar decode hash retained as the bit-exact benchmark oracle.

    ``contexts`` contains the ``ngram_size - 1`` tokens immediately preceding each input id.
    Multiplication and xor are performed modulo 2**64, then interpreted as signed int64 before
    Python's positive-modulus remainder, exactly matching PyTorch int64 arithmetic.
    """
    if len(contexts) != len(input_ids):
        raise ValueError("decode contexts and input ids must have the same batch size")
    if len(layer_multipliers) != ngram_size:
        raise ValueError("one hash multiplier is required per n-gram position")
    expected_heads = (ngram_size - 1) * heads_per_ngram
    if len(vocab_sizes) != expected_heads or len(offsets) != expected_heads:
        raise ValueError(f"expected {expected_heads} hash heads")

    rows: list[list[int]] = []
    sign = 1 << 63
    modulus = 1 << 64
    for context, current in zip(contexts, input_ids):
        if len(context) != ngram_size - 1:
            raise ValueError(f"decode context must contain {ngram_size - 1} tokens")
        tokens = [int(current)]
        crossed_eos = False
        for distance in range(1, ngram_size):
            previous = int(context[-distance])
            crossed_eos = crossed_eos or previous == eos_token_id
            tokens.append(eos_token_id if crossed_eos else previous)

        row: list[int] = []
        for ngram in range(2, ngram_size + 1):
            mixed = (tokens[0] * int(layer_multipliers[0])) & _MASK64
            for position in range(1, ngram):
                mixed ^= (tokens[position] * int(layer_multipliers[position])) & _MASK64
            signed = mixed if mixed < sign else mixed - modulus
            start = (ngram - 2) * heads_per_ngram
            for head in range(start, start + heads_per_ngram):
                row.append(signed % int(vocab_sizes[head]) + int(offsets[head]))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _derive_decode_row_ids_tensor(
    contexts: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    layer_multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    out: torch.Tensor,
    tokens: torch.Tensor,
    mixed: torch.Tensor,
    product: torch.Tensor,
    boundary: torch.Tensor,
    crossed_eos: torch.Tensor,
) -> torch.Tensor:
    """Vectorized signed-int64 decode hash into caller-owned fixed buffers."""
    batch_size = input_ids.numel()
    if contexts.shape != (batch_size, ngram_size - 1):
        raise ValueError(
            f"decode contexts must have shape {(batch_size, ngram_size - 1)}, got "
            f"{tuple(contexts.shape)}"
        )
    expected_heads = (ngram_size - 1) * heads_per_ngram
    if (
        layer_multipliers.numel() != ngram_size
        or vocab_sizes.numel() != expected_heads
        or offsets.numel() != expected_heads
    ):
        raise ValueError(f"expected {expected_heads} hash heads")

    token_window = tokens[:batch_size]
    token_window[:, 0].copy_(input_ids)
    crossed = crossed_eos[:batch_size]
    crossed.zero_()
    is_boundary = boundary[:batch_size]
    for distance in range(1, ngram_size):
        previous = contexts[:, -distance]
        torch.eq(previous, eos_token_id, out=is_boundary)
        crossed.logical_or_(is_boundary)
        token_window[:, distance].copy_(previous)
        token_window[:, distance].masked_fill_(crossed, eos_token_id)

    row_ids = out[:batch_size]
    accumulator = mixed[:batch_size]
    term = product[:batch_size]
    torch.mul(token_window[:, 0], layer_multipliers[0], out=accumulator)
    for position in range(1, ngram_size):
        torch.mul(token_window[:, position], layer_multipliers[position], out=term)
        torch.bitwise_xor(accumulator, term, out=accumulator)
        start = (position - 1) * heads_per_ngram
        stop = start + heads_per_ngram
        torch.remainder(
            accumulator.unsqueeze(1),
            vocab_sizes[start:stop],
            out=row_ids[:, start:stop],
        )
        row_ids[:, start:stop].add_(offsets[start:stop])
    return row_ids


def derive_decode_row_ids_host(
    contexts: Sequence[Sequence[int]] | torch.Tensor,
    input_ids: Sequence[int] | torch.Tensor,
    *,
    layer_multipliers: Sequence[int] | torch.Tensor,
    vocab_sizes: Sequence[int] | torch.Tensor,
    offsets: Sequence[int] | torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Vectorized host hash with PyTorch's signed-int64 overflow and remainder parity."""
    context_tensor = torch.as_tensor(contexts, dtype=torch.int64, device="cpu")
    input_tensor = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
    if context_tensor.ndim != 2 or input_tensor.ndim != 1:
        raise ValueError("decode contexts and input ids must be rank 2 and rank 1")
    batch_size = input_tensor.numel()
    expected_heads = (ngram_size - 1) * heads_per_ngram
    return _derive_decode_row_ids_tensor(
        context_tensor,
        input_tensor,
        layer_multipliers=torch.as_tensor(layer_multipliers, dtype=torch.int64),
        vocab_sizes=torch.as_tensor(vocab_sizes, dtype=torch.int64),
        offsets=torch.as_tensor(offsets, dtype=torch.int64),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=eos_token_id,
        out=torch.empty((batch_size, expected_heads), dtype=torch.int64),
        tokens=torch.empty((batch_size, ngram_size), dtype=torch.int64),
        mixed=torch.empty(batch_size, dtype=torch.int64),
        product=torch.empty(batch_size, dtype=torch.int64),
        boundary=torch.empty(batch_size, dtype=torch.bool),
        crossed_eos=torch.empty(batch_size, dtype=torch.bool),
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def derive_ngram_hash_constants(
    *,
    vocab_size: int,
    ngram_size: int,
    num_ngram_heads: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    seed: int = 1234,
) -> Tuple[List[int], List[int], List[int]]:
    """Recompute (multipliers, per-head vocab sizes, per-head offsets) the way HF derives them at init.

    The checkpoint ships these as int64 tensors, so serving loads them; this is the dummy-weight
    path and the oracle a loader test can check the checkpoint values against.
    """
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    ]
    sizes: List[int] = []
    offsets: List[int] = []
    total = 0
    for head in range(num_ngram_heads):
        global_head = ple_layer_index * num_ngram_heads + head
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return multipliers, sizes, offsets


@dataclass
class PLEMetadata:
    """Per-forward PLE inputs, built once and shared by every PLE layer (sibling of ``FLAMetadata``).

    Frozen contract:
      input_ids      [T] int device -- this forward's tokens, ragged, concatenated in request order
      cu_seqlens     [B+1] int device -- query indptr; decode is ``arange(B+1)``
      seq_lens       host per-request token counts; avoids a device sync in the ragged conv loop
      ngram_context  [B, ngram_size-1] int64 device -- the tokens immediately BEFORE each request's
                     first token of this forward, read from the ``ple_ngram_ctx`` slot state and
                     forced to the boundary (eos) id for fresh rows. The hash never crosses eos,
                     so a fresh sequence passes all-eos.
      state_slots    [B] int64 device -- linear-state slot per request (``Req.linear_slot_idx`` or
                     ``Req.table_idx``); keys every PLE slot state
      fresh_slots    [B] bool device or None -- request starts a new sequence, so read a zero state
      is_decode      one token per request (the batched 4-tap path)
    """

    input_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: Sequence[int]
    ngram_context: torch.Tensor
    state_slots: torch.Tensor
    fresh_slots: torch.Tensor | None
    is_decode: bool


def _state_slot(req) -> int:
    slot = getattr(req, "linear_slot_idx", None)
    return req.table_idx if slot is None else slot


def _ngram_context_pool() -> torch.Tensor:
    pool = get_global_ctx().linear_state_pool
    assert pool is not None and pool.has_slot_state(PLE_NGRAM_STATE), (
        "PLE needs the ple_ngram_ctx slot state (or an explicit context_pool=)"
    )
    return pool.slot_state(PLE_NGRAM_STATE)


def build_ple_metadata(
    batch: Batch,
    args: Qwen4ExpArgs,
    device: torch.device,
    context_pool: torch.Tensor | None = None,
) -> PLEMetadata:
    """Build ``PLEMetadata`` from a scheduler batch.

    The n-gram context is per-request device state (``ple_ngram_ctx`` [num_slots, ngram_size-1],
    rolled forward once per forward by ``commit_ngram_context``), so it never lags the sampled
    token under overlap scheduling and follows the slot on COW/snapshot. A decode batch reads it
    straight off the persistent ``linear_table_idx`` buffer, so the build is capture-safe and
    sync-free. Reuses ``batch.fla_metadata`` (slots / indptr / fresh mask) when the scheduler
    built it.
    """
    reqs = batch.padded_reqs
    ctx_len = args.ngram_size - 1
    eos = args.ngram_boundary_token_id
    if context_pool is None:
        context_pool = _ngram_context_pool()
    assert context_pool.shape[-1] == ctx_len, (
        f"ple_ngram_ctx holds {context_pool.shape[-1]} ids, config wants {ctx_len}"
    )
    fla = getattr(batch, "fla_metadata", None)
    slots_dev = getattr(batch, "linear_table_idx", None)

    if batch.is_decode and slots_dev is not None:
        slots = slots_dev.long()
        bs = slots.numel()
        return PLEMetadata(
            input_ids=batch.input_ids,
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            seq_lens=(1,) * bs,
            ngram_context=context_pool.index_select(0, slots).long(),
            state_slots=slots,
            fresh_slots=None,
            is_decode=True,
        )

    lens = [r.extend_len for r in reqs]
    if fla is not None and fla.has_initial_state is not None:
        cu = fla.cu_seqlens
        slots = fla.cache_indices.long()
        fresh = ~fla.has_initial_state
    else:  # direct-op callers (tests) with no scheduler metadata
        pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}
        cu = torch.tensor([0, *lens], dtype=torch.int64, **pin).cumsum_(0).to(device, non_blocking=True)
        slots = torch.tensor([_state_slot(r) for r in reqs], dtype=torch.int64, **pin).to(device, non_blocking=True)
        fresh = torch.tensor([r.cached_len == 0 for r in reqs], dtype=torch.bool, **pin).to(device, non_blocking=True)
    context = context_pool.index_select(0, slots).long()
    context = torch.where(fresh.unsqueeze(1), context.new_full((), eos), context)
    return PLEMetadata(
        input_ids=batch.input_ids,
        cu_seqlens=cu,
        seq_lens=tuple(lens),
        ngram_context=context,
        state_slots=slots,
        fresh_slots=fresh,
        is_decode=batch.is_decode,
    )


def commit_ngram_context(meta: PLEMetadata, fla, context_pool: torch.Tensor | None = None) -> None:
    """Roll each request's ``ple_ngram_ctx`` forward past this forward's tokens.

    Called ONCE per forward after every PLE layer ran (the layers only read the context);
    also writes the boundary-aligned window to the track slot so a donated snapshot restores
    the context together with the conv state. Pure device arithmetic, capture-safe.
    """
    if context_pool is None:
        context_pool = _ngram_context_pool()
    ids = meta.input_ids.long()
    ctx_len = meta.ngram_context.shape[1]
    steps = torch.arange(ctx_len, device=ids.device)
    if meta.is_decode:
        nxt = torch.cat([meta.ngram_context[:, 1:], ids.view(-1, 1)], dim=1)
    else:
        cu = meta.cu_seqlens.long()
        cand = cu[1:].unsqueeze(1) - ctx_len + steps
        # short extends fall back to the old context: token j of the new window sits at
        # old-context column extend_len + j when it predates this forward
        old = meta.ngram_context.gather(
            1, ((cu[1:] - cu[:-1]).unsqueeze(1) + steps).clamp_(max=ctx_len - 1)
        )
        nxt = torch.where(cand >= cu[:-1].unsqueeze(1), ids[cand.clamp_min(0)], old)
    context_pool.index_copy_(0, meta.state_slots, nxt.to(context_pool.dtype))
    if fla is not None and fla.track_boundary_row is not None:
        win = ids[fla.track_boundary_row.unsqueeze(1) - ctx_len + steps]
        context_pool.index_copy_(0, fla.track_dst, win.to(context_pool.dtype))


class NGramEmbedding(BaseOP):
    """Hashed n-gram lookup: splitmix64 mix of the last n token ids -> per-head prime vocab -> table rows.

    Weight keys (checkpoint names): ``layer_multipliers`` [ngram_size], ``ngram_heads_vocab_sizes``
    and ``ngram_heads_offsets`` [num_ngram_heads], all int64. The table itself is NOT a state-dict
    entry (128 checkpoint shards land in a ``PLETableBackend``); attach it with ``attach_table``.
    """

    def __init__(self, args: Qwen4ExpArgs, table: PLETableBackend | None = None) -> None:
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.num_heads = args.num_ngram_heads
        self.eos_token_id = args.ngram_boundary_token_id
        self.layer_multipliers = torch.empty(args.ngram_size, dtype=torch.int64)
        self.ngram_heads_vocab_sizes = torch.empty(self.num_heads, dtype=torch.int64)
        self.ngram_heads_offsets = torch.empty(self.num_heads, dtype=torch.int64)
        self._table = table
        self._host_hash_constants: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._host_hash_buffers: tuple[torch.Tensor, ...] | None = None

    def attach_table(self, table: PLETableBackend) -> None:
        self._table = table

    def snapshot_host_hash_constants(self, max_batch_size: int | None = None) -> None:
        """Copy constants and allocate the fixed disk-decode hash workspace once."""
        self._host_hash_constants = (
            self.layer_multipliers.detach().cpu().clone(),
            self.ngram_heads_vocab_sizes.detach().cpu().clone(),
            self.ngram_heads_offsets.detach().cpu().clone(),
        )
        if max_batch_size is not None:
            batch_size = int(max_batch_size)
            if batch_size < 1:
                raise ValueError("max PLE decode batch size must be positive")
            self._host_hash_buffers = (
                torch.empty((batch_size, self.num_heads), dtype=torch.int64),
                torch.empty((batch_size, self.ngram_size), dtype=torch.int64),
                torch.empty(batch_size, dtype=torch.int64),
                torch.empty(batch_size, dtype=torch.int64),
                torch.empty(batch_size, dtype=torch.bool),
                torch.empty(batch_size, dtype=torch.bool),
            )

    def host_decode_row_ids(
        self,
        contexts: Sequence[Sequence[int]] | torch.Tensor,
        input_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        if self._host_hash_constants is None:
            raise RuntimeError("PLE host hash constants were not snapshotted after weight load")
        multipliers, sizes, offsets = self._host_hash_constants
        context_tensor = torch.as_tensor(contexts, dtype=torch.int64, device="cpu")
        input_tensor = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        batch_size = input_tensor.numel()
        if self._host_hash_buffers is None:
            self.snapshot_host_hash_constants(batch_size)
        assert self._host_hash_buffers is not None
        if batch_size > self._host_hash_buffers[0].shape[0]:
            raise ValueError(
                f"PLE decode batch {batch_size} exceeds fixed hash buffer "
                f"{self._host_hash_buffers[0].shape[0]}"
            )
        row_ids, tokens, mixed, product, boundary, crossed_eos = self._host_hash_buffers
        return _derive_decode_row_ids_tensor(
            context_tensor,
            input_tensor,
            layer_multipliers=multipliers,
            vocab_sizes=sizes,
            offsets=offsets,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            eos_token_id=self.eos_token_id,
            out=row_ids,
            tokens=tokens,
            mixed=mixed,
            product=product,
            boundary=boundary,
            crossed_eos=crossed_eos,
        )

    @property
    def table(self) -> PLETableBackend:
        assert self._table is not None, "PLE table backend was never attached"
        return self._table

    def _window(self, meta: PLEMetadata):
        """The hash window as ``(packed [B, W], select)``, where ``select`` picks this forward's tokens."""
        ids = meta.input_ids.long()
        ctx_len = self.ngram_size - 1
        if meta.is_decode:
            # a window of exactly ngram_size columns holds every shift the hash can reach
            return torch.cat([meta.ngram_context, ids.view(-1, 1)], dim=1), lambda t: t[:, -1]

        num_reqs = len(meta.seq_lens)
        width = ctx_len + max(meta.seq_lens)
        cu = meta.cu_seqlens.long()
        # Pack the ragged tokens into [B, ctx+max_len] so the shift/boundary logic is one gather.
        flat_pos = torch.arange(ids.numel(), device=ids.device)
        req = (torch.searchsorted(cu, flat_pos, right=True) - 1).clamp_(max=num_reqs - 1)
        col = flat_pos - cu[req] + ctx_len
        packed = ids.new_full((num_reqs, width), self.eos_token_id)
        packed[:, :ctx_len] = meta.ngram_context
        packed[req, col] = ids
        return packed, lambda t: t[req, col]

    def _shift_ignore_eos(self, packed: torch.Tensor) -> List[torch.Tensor]:
        """``out[s][b, p]`` = the token ``s`` places left of ``p``, or eos when the window crosses a boundary."""
        num_reqs, width = packed.shape
        pos = torch.arange(width, device=packed.device)
        eos_pos = torch.where(packed == self.eos_token_id, pos, -1)
        prev_eos = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat([eos_pos.new_full((num_reqs, 1), -1), prev_eos[:, :-1]], dim=1)
        in_segment = pos.unsqueeze(0) - prev_eos - 1

        shifted = [packed]
        for shift in range(1, self.ngram_size):
            src = pos - shift
            gathered = packed.gather(1, src.clamp_min(0).unsqueeze(0).expand(num_reqs, -1))
            valid = (src.unsqueeze(0) >= 0) & (in_segment >= shift)
            shifted.append(torch.where(valid, gathered, packed.new_full((), self.eos_token_id)))
        return shifted

    def row_ids(self, meta: PLEMetadata) -> torch.Tensor:
        """Global table row per (token, hash head): ``[T, num_ngram_heads]`` int64."""
        packed, select = self._window(meta)
        tokens = [select(s) for s in self._shift_ignore_eos(packed)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, tokens[position] * self.layer_multipliers[position])
            head_ids = torch.remainder(mixed.unsqueeze(-1), self.ngram_heads_vocab_sizes[start:end])
            blocks.append(head_ids + self.ngram_heads_offsets[start:end])
        return torch.cat(blocks, dim=-1)

    def forward(self, meta: PLEMetadata, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.table.lookup(self.row_ids(meta), out)


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[width, 1, kernel]`` (key ``conv1d.weight``)."""

    def __init__(self, width: int, kernel: int) -> None:
        self.weight = torch.empty(width, 1, kernel)


def short_conv_reference(
    x: torch.Tensor,
    meta: PLEMetadata,
    states: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Per-request ``F.conv1d`` over ``[state | chunk]``, advancing ``states`` in place.

    Transcription of the HF conv; the shipping paths (one packed conv for prefill, a tap read for
    decode) are diffed against it.
    """
    groups = weight.shape[0]
    state_len = states.shape[-1]
    slots = meta.state_slots
    state = states.index_select(0, slots).to(x.dtype)
    if meta.fresh_slots is not None:
        state = torch.where(meta.fresh_slots.view(-1, 1, 1), torch.zeros_like(state), state)

    outs = []
    new_state = torch.empty_like(state)
    offset = 0
    for i, n in enumerate(meta.seq_lens):
        chunk = x[offset : offset + n].transpose(0, 1).unsqueeze(0)
        history = torch.cat([state[i : i + 1], chunk], dim=-1)
        out = F.conv1d(history, weight, groups=groups, dilation=dilation)
        outs.append(out.squeeze(0).transpose(0, 1))
        new_state[i] = history[0, :, -state_len:]
        offset += n
    states.index_copy_(0, slots, new_state.to(states.dtype))
    return F.silu(torch.cat(outs, dim=0))


class PLELayer(BaseOP):
    """PLE block: hashed n-gram value gated by the residual streams, then a dilated depthwise conv.

    ``forward(R, batch) -> D [T, hc_count*hidden]``; the caller adds ``D`` to ``R`` before the
    attention hyper-connection mix. ``meta`` defaults to ``build_ple_metadata(batch, ...)``;
    ``conv_states`` defaults to ``ctx.linear_state_pool.slot_state("ple_conv", layer_id)`` and is
    ``[num_slots, hc_count*hidden, (ple_conv_kernel_size-1)*ngram_size]`` in the model dtype -- the
    last conv-input columns per request, oldest first. Both are arguments so the reference is
    testable before the pool and the scheduler carry them.

    ``start_prefetch(batch)`` builds the metadata and starts the table gather on the backend's side
    stream; call it at the top of the model forward so the rows land while layer 0 runs, and
    ``forward`` joins it.

    Weight keys (checkpoint names, prefix stripped): ``key_proj.weight`` [hc*hidden, ple_embed_dim],
    ``value_proj.weight`` [hidden, ple_embed_dim], ``norm_key/norm_query/norm_conv.weight``
    [hc*hidden] (zero-centered, loaded RAW), ``conv1d.weight`` [hc*hidden, 1, kernel], plus the
    three ``ple_embedding`` int64 hash buffers.
    """

    def __init__(
        self, config: ModelConfig, layer_id: int, table: PLETableBackend | None = None
    ) -> None:
        args = config.qwen4_args
        self.args = args
        self.layer_id = layer_id
        self.ple_index = args.ple_layer_ids.index(layer_id)
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.dilation = args.ple_conv_dilation
        self.state_len = args.ple_conv_state_len
        width = args.ple_state_width
        self.ple_embedding = NGramEmbedding(args, table)
        self.key_proj = LinearReplicated(args.ple_embed_dim, width, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, args.hidden_size, has_bias=False)
        self.norm_key = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_query = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.norm_conv = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.conv1d = _DepthwiseConv1d(width, args.ple_conv_kernel_size)
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        # the track snapshot gathers the last state_len conv inputs before a xCHUNK boundary; a longer history would reach before the forward's first token
        assert self.state_len <= CHUNK_SIZE, (
            f"PLE conv history {self.state_len} exceeds CHUNK_SIZE {CHUNK_SIZE}"
        )
        self._pending: Tuple[PLEMetadata, torch.Tensor] | None = None

    def start_prefetch(self, batch: Batch, meta: PLEMetadata | None = None) -> None:
        """Hash this forward's n-grams and start the table gather on the side stream."""
        if meta is None:
            meta = build_ple_metadata(batch, self.args, batch.input_ids.device)
        row_ids = self.ple_embedding.row_ids(meta)
        self._pending = (meta, row_ids)
        self.ple_embedding.table.prefetch(row_ids)

    def forward(
        self,
        R: torch.Tensor,
        batch: Batch,
        meta: PLEMetadata | None = None,
        conv_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pending, self._pending = self._pending, None
        row_ids = None
        if meta is None:
            if pending is not None:
                meta, row_ids = pending
            else:
                meta = build_ple_metadata(batch, self.args, R.device)
        elif pending is not None and pending[0] is meta:
            row_ids = pending[1]
        if row_ids is None:
            row_ids = self.ple_embedding.row_ids(meta)

        embeddings = self.ple_embedding.table.lookup(row_ids).to(R.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embeddings))
        value = self.value_proj.forward(embeddings)
        query = self.norm_query.forward(R)
        shape = (-1, self.hc_count, self.hidden_size)
        gate = (key.view(shape) * query.view(shape)).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2)).flatten(-2)
        states = conv_states if conv_states is not None else self._conv_state_slab(R)
        x = self.norm_conv.forward(gated)
        fla = getattr(batch, "fla_metadata", None)
        if fla is not None and fla.track_boundary_row is not None:
            self._write_track_snapshot(states, x, fla)
        return gated + self._short_conv(x, meta, states)

    def _write_track_snapshot(self, states: torch.Tensor, x: torch.Tensor, fla) -> None:
        """Copy the conv history at the GDN track boundary into the same donatable slot, so a radix
        prefix hit restores PLE and GDN state together. Track slots never alias the live slots this
        forward advances, so the two writes are order-independent."""
        src = fla.track_boundary_row.unsqueeze(1) + torch.arange(
            -self.state_len, 0, device=x.device
        )
        window = x[src].transpose(-1, -2).contiguous()
        states.index_copy_(0, fla.track_dst, window.to(states.dtype))

    def _conv_state_slab(self, R: torch.Tensor) -> torch.Tensor:
        pool = get_global_ctx().linear_state_pool
        assert pool is not None, "PLE needs ctx.linear_state_pool or an explicit conv_states"
        assert pool.has_slot_state(PLE_CONV_STATE), (
            "ModelConfig.slot_states does not declare the PLE conv history"
        )
        return pool.slot_state(PLE_CONV_STATE, self.layer_id)

    def _read_state(
        self, meta: PLEMetadata, states: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        state = states.index_select(0, meta.state_slots).to(dtype)
        if meta.fresh_slots is not None:
            state = torch.where(meta.fresh_slots.view(-1, 1, 1), torch.zeros_like(state), state)
        return state

    def _short_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """silu of the dilated depthwise conv over [state | x], and roll the per-request state."""
        if meta.is_decode:
            return self._decode_conv(x, meta, states)
        return self._prefill_conv(x, meta, states)

    def _decode_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """Batched tap read: taps t-9, t-6, t-3 come off the state slab, tap t from this token."""
        state = self._read_state(meta, states, x.dtype)
        column = x.unsqueeze(-1)
        # fp32 products, like the conv1d the prefill path runs
        window = torch.cat([state[..., :: self.dilation], column], dim=-1).float()
        out = (window * self.conv1d.weight.squeeze(1).float()).sum(-1)
        states.index_copy_(
            0, meta.state_slots, torch.cat([state[..., 1:], column], dim=-1).to(states.dtype)
        )
        return F.silu(out.to(x.dtype))

    def _prefill_conv(
        self, x: torch.Tensor, meta: PLEMetadata, states: torch.Tensor
    ) -> torch.Tensor:
        """One conv over every request packed as ``[state_0 | chunk_0 | state_1 | chunk_1 | ...]``.

        The blocks abut exactly, so each output window stays inside its own request: request i's
        first token reads history columns base_i .. base_i+state_len, which is its own state.
        """
        lens = list(meta.seq_lens)
        num_reqs, width = len(lens), x.shape[1]
        out_index, state_index, next_state_index = self._prefill_indices(lens, x.device)

        state = self._read_state(meta, states, x.dtype)
        history = x.new_empty(width, x.shape[0] + num_reqs * self.state_len)
        history.index_copy_(1, state_index, state.permute(1, 0, 2).reshape(width, -1))
        history.index_copy_(1, out_index + self.state_len, x.transpose(0, 1).contiguous())

        out = F.conv1d(
            history.unsqueeze(0), self.conv1d.weight, groups=width, dilation=self.dilation
        ).squeeze(0)
        new_state = history.index_select(1, next_state_index).view(width, num_reqs, self.state_len)
        states.index_copy_(
            0, meta.state_slots, new_state.permute(1, 0, 2).to(states.dtype).contiguous()
        )
        return F.silu(out.index_select(1, out_index).transpose(0, 1))

    def _prefill_indices(self, lens: List[int], device: torch.device):
        """Columns of the packed history: this forward's outputs, the state block, the next state block."""
        state_len = self.state_len
        counts = torch.tensor(lens, dtype=torch.int64)
        cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        pad = torch.arange(len(lens), dtype=torch.int64) * state_len
        base = cu[:-1] + pad
        out_index = torch.arange(int(cu[-1])) + torch.repeat_interleave(pad, counts)
        span = torch.arange(state_len, dtype=torch.int64)
        packed = torch.cat(
            [
                out_index,
                (base.unsqueeze(1) + span).reshape(-1),
                ((base + counts).unsqueeze(1) + span).reshape(-1),
            ]
        )
        if torch.cuda.is_available():
            packed = packed.pin_memory()
        packed = packed.to(device, non_blocking=True)
        n_out, n_state = out_index.numel(), len(lens) * state_len
        return packed[:n_out], packed[n_out : n_out + n_state], packed[n_out + n_state :]


__all__ = [
    "DiskStagedTable",
    "GpuResidentTable",
    "HMMMappedTable",
    "NGramEmbedding",
    "ZeroTable",
    "PLELayer",
    "PLEMetadata",
    "PLETableBackend",
    "PinnedUVATable",
    "build_ple_metadata",
    "derive_decode_row_ids_host",
    "derive_ngram_hash_constants",
    "hmm_row_address",
    "process_major_faults",
    "short_conv_reference",
]
