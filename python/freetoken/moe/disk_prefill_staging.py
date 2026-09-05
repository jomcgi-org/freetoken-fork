"""Experimental bounded file-to-GPU transport for uncaptured prefill.

The staged DISK policy reuses one instance across banks, layers, and requests
without allocating a whole-layer mirror.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.host_banks import _BLK, _preadv_all, coalesced_row_ranges

_RESIDENCY_BITS = bytes(value & 1 for value in range(256))


class DiskPrefillStaging:
    """Read exact file ranges into two pinned buffers and copy unchanged bytes.

    Copies use the caller's current CUDA stream. The next GEMM on that stream
    observes all transfers. Before refilling a staging buffer, a host wait for
    its completion event prevents overwriting bytes still owned by DMA.
    Call ``synchronize`` before discarding this object. One scheduler thread
    owns an instance; concurrent calls are unsupported.
    """

    def __init__(
        self, device: torch.device, *, chunk_bytes: int = 32 << 20,
        direct_io: bool = False, reuse_cached_rows: bool = False,
    ):
        self.device = torch.device(device)
        chunk_bytes = int(chunk_bytes)
        if self.device.type != "cuda" or chunk_bytes <= 0:
            raise ValueError("staging requires a CUDA device and a positive chunk size")
        if direct_io and (not hasattr(os, "O_DIRECT") or chunk_bytes < 2 * _BLK):
            raise ValueError("direct staging requires O_DIRECT and at least two alignment blocks")
        if reuse_cached_rows and not direct_io:
            raise ValueError("cached-row reuse requires the direct staging reader")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.chunk_bytes = chunk_bytes
        self.direct_io = direct_io
        self.reuse_cached_rows = reuse_cached_rows
        if reuse_cached_rows:
            self._probe_bytes = min(chunk_bytes, 32 << 20)
            self._residency = bytearray(self._probe_bytes // mmap.PAGESIZE + 2)
            self._residency_pointer = ctypes.c_void_p(
                ctypes.addressof(ctypes.c_char.from_buffer(self._residency))
            )
            self._mincore = ctypes.CDLL(None, use_errno=True).mincore
            self._mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            self._mincore.restype = ctypes.c_int
        self._allocations = [
            alloc_pinned_tensor(self.chunk_bytes, dtype=torch.uint8) for _ in range(2)
        ]
        self.buffers = self._allocations
        if direct_io:
            # Align within the existing allocation. This spends a little usable
            # capacity instead of adding memory outside the governor's ring budget.
            self.buffers = []
            for allocation in self._allocations:
                offset = (-allocation.data_ptr()) % _BLK
                usable = (allocation.numel() - offset) // _BLK * _BLK
                self.buffers.append(allocation[offset:offset + usable])
        self._views = [memoryview(buffer.numpy()).cast("B") for buffer in self.buffers]
        self._events = [torch.cuda.Event() for _ in self.buffers]
        self._pending = [False] * len(self.buffers)
        self._next = 0

    @property
    def pinned_bytes(self) -> int:
        return sum(buffer.numel() for buffer in self._allocations)

    def synchronize(self) -> None:
        for slot, event in enumerate(self._events):
            if self._pending[slot]:
                event.synchronize()
                self._pending[slot] = False

    def _residency_snapshot(self, address: int, length: int) -> tuple[bytes | None, int]:
        """Bounded, advisory page state. Failure must select a real file read."""
        head = address % mmap.PAGESIZE
        pages = (head + length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        if length > self._probe_bytes or pages > len(self._residency):
            raise ValueError("residency probe exceeds its fixed metadata buffer")
        if self._mincore(address - head, head + length, self._residency_pointer):
            return None, head
        # Only bit zero is defined. Copy the bounded snapshot before reusing
        # its output buffer for another group of rows.
        return memoryview(self._residency)[:pages].tobytes().translate(_RESIDENCY_BITS), head

    def _cached_row_ranges(self, source: torch.Tensor, ranges):
        """Coalesce selected rows by their advisory source, never by route priority."""
        row_bytes = source.stride(0) * source.element_size()
        nbytes = source.numel() * source.element_size()
        group_bytes = max(1, self._probe_bytes // row_bytes) * row_bytes
        previous_group = None
        snapshot = None
        pending = None
        for start, length in ranges:
            for row_start in range(start, start + length, row_bytes):
                if row_bytes > self._probe_bytes:
                    # Unusually large rows still use bounded metadata and are
                    # cached only if every piece is resident.
                    cached = True
                    for offset in range(0, row_bytes, self._probe_bytes):
                        bits, _head = self._residency_snapshot(
                            source.data_ptr() + row_start + offset,
                            min(self._probe_bytes, row_bytes - offset),
                        )
                        if bits is None or b"\0" in bits:
                            cached = False
                            break
                else:
                    group = row_start // group_bytes * group_bytes
                    if group != previous_group:
                        snapshot, head = self._residency_snapshot(
                            source.data_ptr() + group, min(group_bytes, nbytes - group),
                        )
                        previous_group = group
                    lo = (head + row_start - group) // mmap.PAGESIZE
                    hi = (head + row_start - group + row_bytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE
                    cached = snapshot is not None and snapshot.find(b"\0", lo, hi) < 0
                if pending is not None:
                    prior_start, prior_length, prior_cached = pending
                    if prior_cached == cached and prior_start + prior_length == row_start:
                        pending = (prior_start, prior_length + row_bytes, cached)
                        continue
                    yield pending
                pending = (row_start, row_bytes, cached)
        if pending is not None:
            yield pending

    def copy_bank(self, source: torch.Tensor, destination: torch.Tensor, rows=None) -> int:
        """Copy all rows, or the exact valid row union, to original row positions.

        Source tensors must retain their authoritative file-backed HostBank.
        Packed weights and all scale formats travel as bytes, without casting.
        The return value is the logical byte count, without a device readback.
        Experimental direct I/O rounds file reads outward to alignment blocks,
        but copies only the requested bytes. I/O errors propagate without an
        application fallback. The experiment must validate filesystem support
        separately because the kernel can fall back for unsupported inodes.
        It must not run across a concurrent fork; the serving worker owns the
        ring after spawning.
        """
        bank = getattr(source, "_freetoken_host_bank", None)
        if bank is None or not bank._disk or bank._uffd or bank._file_path is None:
            raise ValueError("source must be an ordinary file-backed HostBank tensor")
        if (
            not source.is_contiguous() or not destination.is_contiguous()
            or source.shape != destination.shape or source.dtype != destination.dtype
            or not destination.is_cuda or destination.device != self.device
            or source.ndim == 0
        ):
            raise ValueError("source and destination need matching contiguous bank geometry")
        source_offset = source.data_ptr() - bank.tensor.data_ptr()
        nbytes = source.numel() * source.element_size()
        if source_offset < 0 or source_offset + nbytes > bank.nbytes:
            raise ValueError("source view extends outside its backing bank")
        file_offset = bank._map_offset + bank._view_offset + source_offset
        ranges = (
            [(0, nbytes)] if rows is None else
            coalesced_row_ranges(rows, source.stride(0) * source.element_size(), limit=nbytes)
        )
        if not ranges or nbytes == 0:
            return 0
        target = destination.view(torch.uint8).view(-1)
        stream = torch.cuda.current_stream(self.device)
        copied = 0
        flags = os.O_RDONLY | (os.O_DIRECT if self.direct_io else 0)
        fd = os.open(bank._file_path, flags)
        cached_fd = None
        try:
            # mincore can conceal page state behind an all-resident answer for
            # unprivileged readers. Conservatively use it only for owned files.
            reuse = self.reuse_cached_rows and os.fstat(fd).st_uid == os.geteuid()
            planned = (
                self._cached_row_ranges(source, ranges) if reuse else
                ((start, length, False) for start, length in ranges)
            )
            for start, length, cached in planned:
                if cached and cached_fd is None:
                    cached_fd = os.open(bank._file_path, os.O_RDONLY)
                    os.posix_fadvise(cached_fd, 0, 0, os.POSIX_FADV_RANDOM)
                read_fd = cached_fd if cached else fd
                direct = self.direct_io and not cached
                done = 0
                while done < length:
                    slot = self._next
                    if self._pending[slot]:
                        self._events[slot].synchronize()
                        self._pending[slot] = False
                    offset = file_offset + start + done
                    head = offset % _BLK if direct else 0
                    count = min(self.buffers[slot].numel() - head, length - done)
                    if direct:
                        span = (head + count + _BLK - 1) // _BLK * _BLK
                        _preadv_all(read_fd, self._views[slot][:span], offset - head, head + count)
                    else:
                        filled = 0
                        while filled < count:
                            got = os.preadv(
                                read_fd, [self._views[slot][filled:count]], offset + filled,
                            )
                            if got <= 0:
                                raise OSError("short file read while staging prefill weights")
                            filled += got
                    target[start + done:start + done + count].copy_(
                        self.buffers[slot][head:head + count], non_blocking=True,
                    )
                    self._events[slot].record(stream)
                    self._pending[slot] = True
                    self._next = (slot + 1) % len(self.buffers)
                    done += count
                    copied += count
        finally:
            if cached_fd is not None:
                os.close(cached_fd)
            os.close(fd)
        return copied
