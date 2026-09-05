"""Experimental bounded file-to-GPU transport for uncaptured prefill.

The staged DISK policy reuses one instance across banks, layers, and requests
without allocating a whole-layer mirror.
"""

from __future__ import annotations

import os

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.host_banks import _BLK, _preadv_all, coalesced_row_ranges


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
        direct_io: bool = False,
    ):
        self.device = torch.device(device)
        chunk_bytes = int(chunk_bytes)
        if self.device.type != "cuda" or chunk_bytes <= 0:
            raise ValueError("staging requires a CUDA device and a positive chunk size")
        if direct_io and (not hasattr(os, "O_DIRECT") or chunk_bytes < 2 * _BLK):
            raise ValueError("direct staging requires O_DIRECT and at least two alignment blocks")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.chunk_bytes = chunk_bytes
        self.direct_io = direct_io
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
        try:
            for start, length in ranges:
                done = 0
                while done < length:
                    slot = self._next
                    if self._pending[slot]:
                        self._events[slot].synchronize()
                        self._pending[slot] = False
                    offset = file_offset + start + done
                    head = offset % _BLK if self.direct_io else 0
                    count = min(self.buffers[slot].numel() - head, length - done)
                    if self.direct_io:
                        span = (head + count + _BLK - 1) // _BLK * _BLK
                        _preadv_all(fd, self._views[slot][:span], offset - head, head + count)
                    else:
                        filled = 0
                        while filled < count:
                            got = os.preadv(
                                fd, [self._views[slot][filled:count]], offset + filled,
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
            os.close(fd)
        return copied
