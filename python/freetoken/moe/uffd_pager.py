"""Linux userfaultfd pager for file-backed expert-bank rows.

The native extension owns userfaultfd, io_uring, the handler thread, and the
global row LRU.  This module intentionally imports that extension only after a
Linux check so ordinary imports and checkpoint tooling remain portable.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


_UFFD_REQUIREMENT = (
    "UFFD expert paging requires Linux userfaultfd access. Set "
    "/proc/sys/vm/unprivileged_userfaultfd=1 or grant CAP_SYS_PTRACE to the "
    "serving process."
)


def probe_uffd_support(*, _native_module=None) -> None:
    """Probe the kernel permission and UFFD missing-page API at startup."""
    if sys.platform != "linux":
        raise RuntimeError(
            "--moe-disk-pager uffd is Linux-only; use --moe-disk-pager madvise "
            "on this platform"
        )
    try:
        native = _native_module
        if native is None:
            from freetoken.kernel import _uffd_pager as native
        native.probe()
    except ImportError as exc:
        raise RuntimeError(
            "the Linux UFFD pager extension is not built; rebuild FreeToken or use "
            "--moe-disk-pager madvise"
        ) from exc
    except (OSError, RuntimeError) as exc:
        setting = "unknown"
        try:
            setting = Path("/proc/sys/vm/unprivileged_userfaultfd").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            pass
        raise RuntimeError(
            f"{_UFFD_REQUIREMENT} Current vm.unprivileged_userfaultfd={setting!r}. "
            f"Kernel probe failed: {exc}"
        ) from exc


class UFFDPager:
    """One native pager and byte-budgeted LRU shared by all DISK bank regions."""

    backend = "uffd"

    def __init__(self, budget_bytes: int, *, _native_module=None) -> None:
        budget_bytes = int(budget_bytes)
        if budget_bytes <= 0:
            raise ValueError("UFFD pager budget must be positive")
        probe_uffd_support(_native_module=_native_module)
        native = _native_module
        if native is None:
            from freetoken.kernel import _uffd_pager as native
        self._native = native.UffdPager(budget_bytes)
        self.budget_bytes = budget_bytes

    def register_bank(
        self,
        bank,
        *,
        file_path: str,
        file_offset: int,
        row_bytes: int,
        num_rows: int,
    ) -> int:
        """Register an anonymous bank mapping and its aligned FTW source rows."""
        region = self._native.add_region(
            bank.addr,
            len(bank._buf),
            bank.nbytes,
            str(file_path),
            int(file_offset),
            int(row_bytes),
            int(num_rows),
        )
        return int(region)

    def prefetch(self, banks, expert_ids) -> int:
        """Synchronously materialize missing rows before CPU expert compute.

        The native request protects the full union across all bank tensors from
        eviction while it is being installed. The return value remains a page
        count for compatibility with the existing DISK telemetry.
        """
        regions = [int(bank._pager_region) for bank in banks]
        rows = sorted({int(row) for row in expert_ids if int(row) >= 0})
        if not regions or not rows:
            return 0
        return int(self._native.prefetch(regions, rows))

    def is_resident(self, bank, expert_id: int) -> bool:
        """Read the native residency bitmap for one bank row."""
        return bool(self._native.is_resident(bank._pager_region, int(expert_id)))

    def validate_working_set(self, banks, num_rows: int, *, context: str) -> None:
        """Reject a route union that cannot remain resident through compute."""
        row_bytes = sum(bank.nbytes // bank.tensor.shape[0] for bank in banks)
        required = row_bytes * int(num_rows)
        if required > self.budget_bytes:
            raise ValueError(
                f"UFFD pager budget {self.budget_bytes / 2**30:.2f} GiB cannot hold "
                f"the {required / 2**30:.2f} GiB {context} expert-row working set; "
                "increase --moe-pager-budget-gib"
            )

    def stats(self, *, reset: bool = False) -> dict:
        """Return fill, eviction, residency, and fill-latency counters."""
        return dict(self._native.stats(bool(reset)))

    def raise_if_error(self) -> None:
        """Surface an asynchronous fault-handler failure on the serving thread."""
        self._native.raise_if_error()


def make_uffd_pager(budget_gib: float, *, _native_module=None) -> UFFDPager:
    """Validate a GiB budget and construct the startup-probed pager."""
    budget_gib = float(budget_gib)
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("--moe-pager-budget-gib must be a finite positive number")
    return UFFDPager(int(budget_gib * (1 << 30)), _native_module=_native_module)


__all__ = ["UFFDPager", "make_uffd_pager", "probe_uffd_support"]
