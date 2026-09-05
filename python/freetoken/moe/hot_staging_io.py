"""Optional buffered file reads into the existing HOT staging allocation."""

from __future__ import annotations

import os

import torch


class HotRowFileReader:
    """Read authoritative row bytes without a file-mapped tensor copy.

    One HOT staging call owns this reader and closes its descriptors on exit.
    The caller retains the pinned tensors, cancellation boundaries, DMA waits,
    and publication protocol. No new weight buffer is allocated here.
    """

    def __init__(self):
        if not hasattr(os, "preadv"):
            raise RuntimeError("buffered HOT staging requires os.preadv")
        self._fds = {}

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def copy_row(self, source: torch.Tensor, row: int, destination: torch.Tensor) -> bool:
        """Return False for banks that keep their original tensor-copy path."""
        bank = getattr(source, "_freetoken_host_bank", None)
        if (
            bank is None or not bank._disk or bank._uffd
            or getattr(bank, "_tmpfs_backed", False)
        ):
            return False
        if bank._file_path is None:
            raise ValueError("HOT file source has no backing path")
        if (
            source.device.type != "cpu" or destination.device.type != "cpu"
            or source.ndim == 0 or not source.is_contiguous()
            or not destination.is_contiguous()
            or destination.shape != source.shape[1:]
            or destination.dtype != source.dtype
        ):
            raise ValueError("HOT file staging requires matching contiguous CPU row geometry")
        if row < 0 or row >= source.shape[0]:
            raise ValueError("HOT file row is outside its source view")
        source_offset = source.data_ptr() - bank.tensor.data_ptr()
        if source_offset < 0 or source_offset + source.numel() * source.element_size() > bank.nbytes:
            raise ValueError("HOT source view extends outside its backing bank")
        count = destination.numel() * destination.element_size()
        if not count:
            return True
        offset = (
            bank._map_offset + bank._view_offset + source_offset
            + row * source.stride(0) * source.element_size()
        )
        fd = self._fds.get(bank._file_path)
        if fd is None:
            fd = os.open(bank._file_path, os.O_RDONLY)
            self._fds[bank._file_path] = fd
        # Flatten first so per-expert scalar scales also expose their bytes.
        target = memoryview(destination.reshape(-1).view(torch.uint8).numpy()).cast("B")
        done = 0
        while done < count:
            got = os.preadv(fd, [target[done:]], offset + done)
            if got <= 0:
                raise OSError("short file read while staging HOT row")
            done += got
        return True
