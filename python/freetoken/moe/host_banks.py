"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import mmap
import os
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can directly feed GPU movement. LOCKED and
    PAGEABLE layers decode on the CPU executor. File-backed DISK layers either do the
    same or copy routed misses through the bounded pinned GPU-fetch ring.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"
    DISK = "disk"


_DEFAULT_CHUNK = 8 << 20

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []

def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.
    * ``"file"`` -- a read-only shared mapping of one FTW bank entry. It stays
      DISK resident and applies ``MADV_RANDOM`` at creation.
    * ``"uffd"`` -- a writable anonymous mapping registered with the process-wide
      UFFD pager. Its FTW pages are installed on demand and governed by userspace LRU.

    The buffer is rounded up to its backing alignment; ``tensor`` views exactly
    ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = (
        "tensor", "addr", "nbytes", "_buf", "_pinned", "_locked", "_disk",
        "_view_offset", "_uffd", "_pager", "_pager_region", "_file_path",
        "_map_offset",
    )

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 *, backing: str | None = None, file_path: str | None = None,
                 file_offset: int = 0, disk_pager=None):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda", "file", "uffd"), backing
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        allocation_alignment = mmap.PAGESIZE if backing == "uffd" else _BLK
        asize = (
            (self.nbytes + allocation_alignment - 1) // allocation_alignment
        ) * allocation_alignment
        self._disk = backing in ("file", "uffd")
        self._uffd = backing == "uffd"
        self._pager = disk_pager
        self._pager_region = None
        self._file_path = file_path
        self._map_offset = 0
        self._view_offset = 0
        if backing == "file":
            if file_path is None:
                raise ValueError("file-backed HostBank requires file_path")
            map_off = file_offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            self._map_offset = map_off
            self._view_offset = file_offset - map_off
            # Safetensors tensor payloads are not necessarily page aligned. Map from the
            # aligned floor, then expose the tensor at its byte offset within that mapping.
            # Do not round the mapping past the payload because the tensor may end at EOF.
            map_len = self._view_offset + self.nbytes
            fd = os.open(file_path, os.O_RDONLY)
            try:
                self._buf = mmap.mmap(
                    fd, map_len, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ,
                    offset=map_off,
                )
            finally:
                os.close(fd)
            try:
                self._buf.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError):
                pass
            _LIVE_BUFFERS.append(self._buf)
            self._pinned = False
        elif backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._buf = raw.numpy()[off:off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
        else:
            if self._uffd:
                self._buf = mmap.mmap(
                    -1, asize, flags=mmap.MAP_PRIVATE,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            else:
                self._buf = mmap.mmap(-1, asize)
            # Both are lazy anonymous address space; pages materialize only on fill.
            _LIVE_BUFFERS.append(self._buf)
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            self._pinned = False
            if self._uffd:
                if file_path is None or disk_pager is None:
                    raise ValueError(
                        "UFFD-backed HostBank requires file_path and disk_pager"
                    )
                if not shape or shape[0] <= 0 or self.nbytes % shape[0]:
                    raise ValueError("UFFD expert bank must have a non-empty row axis")
                self._pager_region = disk_pager.register_bank(
                    self,
                    file_path=file_path,
                    file_offset=file_offset,
                    row_bytes=self.nbytes // shape[0],
                    num_rows=shape[0],
                )
        with warnings.catch_warnings():
            if backing == "file":
                warnings.filterwarnings(
                    "ignore", message="The given buffer is not writable",
                    category=UserWarning,
                )
            self.tensor = torch.frombuffer(
                self._buf, dtype=dtype, count=self.nbytes // elsize,
                offset=self._view_offset,
            ).view(*shape)
        if self._disk:
            self.addr = self.tensor.data_ptr()
        # The CPU executor only receives tensors. Keep the owning mapping reachable from
        # each direct FTW view so it can issue expert-granular prefetches.
        self.tensor._freetoken_host_bank = self
        self._locked = False

    @property
    def residency(self) -> HostResidency:
        if self._disk:
            return HostResidency.DISK
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._disk:
            raise RuntimeError("a read-only file-backed DISK bank cannot be CUDA-pinned")
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._pinned or self._uffd:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._disk:
            raise RuntimeError("a file-backed DISK bank cannot be mlock'd")
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self.addr, len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True

    def prefetch_rows(self, row_ids) -> int:
        """Issue one coalesced ``MADV_WILLNEED`` sweep for selected rows.

        Returns the number of distinct 4 KiB pages requested. Non-DISK banks are a
        no-op so callers can walk a layer's full bank schema without branching.
        """
        if not self._disk:
            return 0
        if self._uffd:
            return self._pager.prefetch([self], row_ids)
        stride = self.tensor.stride(0) * self.tensor.element_size()
        ranges = coalesced_page_ranges(
            row_ids, stride, limit=self.nbytes, page_offset=self._view_offset,
        )
        pages = 0
        for advise_start, length in ranges:
            advise_end = min(len(self._buf), advise_start + length)
            self._buf.madvise(
                mmap.MADV_WILLNEED, advise_start, advise_end - advise_start,
            )
            pages += (advise_end - advise_start + _BLK - 1) // _BLK
        return pages

    def prefetch_experts(self, expert_ids) -> int:
        """Compatibility name for expert-bank row prefetching."""
        return self.prefetch_rows(expert_ids)

    def populate_rows(self, row_ids, scratch) -> int:
        """Read selected file-backed rows through one reusable bounded buffer.

        The data is deliberately discarded. Reading the backing file populates the
        page cache so the executor's fixed mmap pointers encounter minor faults.
        UFFD banks stay on their pager-owned prefetch path.
        """
        if not self._disk or self._uffd:
            return 0
        if self._file_path is None:
            raise OSError("file-backed bank has no source path")
        dst = memoryview(scratch).cast("B")
        if not dst:
            raise ValueError("populate scratch buffer must not be empty")
        stride = self.tensor.stride(0) * self.tensor.element_size()
        ranges = coalesced_row_ranges(
            row_ids,
            stride,
            limit=self.nbytes,
            base_offset=self._map_offset + self._view_offset,
        )
        if not ranges:
            return 0
        fd = os.open(self._file_path, os.O_RDONLY)
        total = 0
        try:
            for start, length in ranges:
                done = 0
                while done < length:
                    want = min(len(dst), length - done)
                    if hasattr(os, "preadv"):
                        got = os.preadv(fd, [dst[:want]], start + done)
                    else:
                        data = os.pread(fd, want, start + done)
                        got = len(data)
                        dst[:got] = data
                    if got <= 0:
                        raise OSError(
                            f"short populate read: {done} of {length} bytes at {start}"
                        )
                    done += got
                    total += got
        finally:
            os.close(fd)
        return total

    def populate_experts(self, expert_ids, scratch) -> int:
        """Compatibility name for expert-bank row population."""
        return self.populate_rows(expert_ids, scratch)

    def release_rows(self, row_ids) -> int:
        """Mark one-pass DISK rows as non-reusable after prefill.

        File-backed mappings use ``POSIX_FADV_NOREUSE`` when available. Unlike an
        unconditional ``MADV_DONTNEED``, this does not discard pages that were already
        part of the decode working set before prefill. Platforms without NOREUSE fall
        back to DONTNEED on the same page-deduplicated ranges. UFFD residency is owned
        by its bounded native LRU and is left to that pager.
        """
        if not self._disk or self._uffd:
            return 0
        stride = self.tensor.stride(0) * self.tensor.element_size()
        ranges = coalesced_page_ranges(
            row_ids, stride, limit=self.nbytes, page_offset=self._view_offset,
        )
        if not ranges:
            return 0
        advice = getattr(os, "POSIX_FADV_NOREUSE", None)
        if advice is not None and self._file_path is not None:
            fd = os.open(self._file_path, os.O_RDONLY)
            try:
                for start, length in ranges:
                    os.posix_fadvise(
                        fd, self._map_offset + start, length, advice,
                    )
            finally:
                os.close(fd)
        else:
            for start, length in ranges:
                end = min(len(self._buf), start + length)
                self._buf.madvise(mmap.MADV_DONTNEED, start, end - start)
        return sum((length + _BLK - 1) // _BLK for _start, length in ranges)


def coalesced_page_ranges(
    expert_ids,
    expert_stride: int,
    *,
    limit: int | None = None,
    page_size: int = _BLK,
    page_offset: int = 0,
) -> list[tuple[int, int]]:
    """Map expert rows to deduplicated, adjacent-coalesced page ranges.

    Negative route sentinels are ignored. ``limit`` clips the final page range to the
    mapping's page-rounded length while still returning page-aligned lengths.
    """
    if expert_stride <= 0 or page_size <= 0:
        raise ValueError("expert_stride and page_size must be positive")
    pages: set[int] = set()
    for raw in expert_ids:
        expert_id = int(raw)
        if expert_id < 0:
            continue
        lo = expert_id * expert_stride
        hi = lo + expert_stride
        if limit is not None and (lo >= limit or hi > limit):
            raise ValueError(f"expert id {expert_id} exceeds bank size {limit}")
        lo += page_offset
        hi += page_offset
        pages.update(range(lo // page_size, (hi + page_size - 1) // page_size))
    if not pages:
        return []
    ordered = sorted(pages)
    out: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        out.append((start * page_size, (prev - start + 1) * page_size))
        start = prev = page
    out.append((start * page_size, (prev - start + 1) * page_size))
    return out


def coalesced_row_ranges(
    expert_ids,
    expert_stride: int,
    *,
    limit: int | None = None,
    base_offset: int = 0,
) -> list[tuple[int, int]]:
    """Return sorted, deduplicated exact row ranges, joining adjacent rows.

    Offsets include ``base_offset`` and are suitable for ``preadv`` against the
    bank's backing file. Unlike :func:`coalesced_page_ranges`, no unrelated bytes
    sharing a page are included.
    """
    if expert_stride <= 0 or base_offset < 0:
        raise ValueError("expert_stride must be positive and base_offset non-negative")
    rows = sorted({int(raw) for raw in expert_ids if int(raw) >= 0})
    out: list[tuple[int, int]] = []
    for expert_id in rows:
        start = expert_id * expert_stride
        end = start + expert_stride
        if limit is not None and (start >= limit or end > limit):
            raise ValueError(f"expert id {expert_id} exceeds bank size {limit}")
        file_start = base_offset + start
        if out and out[-1][0] + out[-1][1] == file_start:
            previous_start, previous_length = out[-1]
            out[-1] = (previous_start, previous_length + expert_stride)
        else:
            out.append((file_start, expert_stride))
    return out


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too


def _os_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], num_layers: int
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name."""
    return {
        name: [HostBank(shape, dtype) for _ in range(num_layers)]
        for name, (shape, dtype) in specs.items()
    }


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = ("labels", "applied", "has_unpinned", "actual")

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


@contextlib.contextmanager
def requested_residency(labels: list[str] | None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins)."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def _settle(bank: HostBank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()
    elif residency == HostResidency.DISK.value and bank.residency is not HostResidency.DISK:
        raise RuntimeError("DISK residency requires an FTW pager-backed HostBank")


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin."""
    plan = _requested_residency
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_id, layer_bank in enumerate(bank):
                residency = (
                    HostResidency.PINNED.value if plan is None
                    else plan.residency_for(layer_id)
                )
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
        else:
            bank.pin()


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            bank, residency, plan, layer_id = item
            try:
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank, residency: str = HostResidency.PINNED.value,
               plan=None, layer_id: int | None = None) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


def _preadv_all(fd: int, dst: memoryview, offset: int, need: int) -> None:
    """preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.
    """
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, file_offset, nbytes, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "born_pinned_default",
    "coalesced_page_ranges",
    "coalesced_row_ranges",
    "pin_banks",
    "read_file_into",
    "read_range_into",
    "requested_residency",
]
