"""Bounded advisory probes for skipping redundant CPU populate reads."""

import ctypes
import mmap
import os
import sys


_RESIDENCY_BITS = bytes(value & 1 for value in range(256))
_PROBE_BYTES = 32 << 20


class ResidentRangeProbe:
    """One populate call owns this small buffer, including background callers.

    A resident answer is only a hint. Pages can leave RAM before computation;
    the original file mapping must remain authoritative and able to demand-fault.
    """

    def __init__(self, max_bytes):
        self.max_bytes = min(int(max_bytes), _PROBE_BYTES)
        if self.max_bytes <= 0:
            raise ValueError('residency probe needs a positive byte bound')
        self._bits = bytearray((self.max_bytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE + 1)
        self._pointer = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._bits)))
        self._mincore = ctypes.CDLL(None, use_errno=True).mincore
        self._mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        self._mincore.restype = ctypes.c_int

    def resident(self, address, length):
        """True only when every page covering the requested bytes is resident."""
        if address <= 0 or length <= 0:
            return False
        while length:
            count = min(length, self.max_bytes)
            head = address % mmap.PAGESIZE
            pages = (head + count + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            if self._mincore(address - head, head + count, self._pointer):
                return False
            bits = memoryview(self._bits)[:pages].tobytes().translate(_RESIDENCY_BITS)
            if b'\0' in bits:
                return False
            address += count
            length -= count
        return True


def owned_file_probe(fd, max_bytes):
    """Avoid concealed mincore answers for files owned by another user."""
    if sys.platform != 'linux' or os.fstat(fd).st_uid != os.geteuid():
        return None
    try:
        return ResidentRangeProbe(max_bytes)
    except (AttributeError, OSError):
        return None
