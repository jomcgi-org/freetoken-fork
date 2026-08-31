"""Real userfaultfd and io_uring coverage, collected only on Linux."""

from __future__ import annotations

import sys

import pytest
import torch


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux userfaultfd test")


def test_uffd_row_fill_bitmap_lru_and_fault_path(tmp_path):
    try:
        from freetoken.kernel import _uffd_pager
    except ImportError:
        pytest.skip("UFFD pager extension is not built")

    try:
        _uffd_pager.probe()
        native = _uffd_pager.UffdPager(2 * 4096)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"kernel does not permit the UFFD/io_uring test: {exc}")

    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.uffd_pager import UFFDPager

    source = tmp_path / "rows.ftw"
    source.write_bytes(b"A" * 4096 + b"B" * 4096 + b"C" * 4096)
    wrapper = object.__new__(UFFDPager)
    wrapper._native = native
    wrapper.budget_bytes = 2 * 4096
    bank = HostBank(
        (3, 4096), torch.uint8, backing="uffd",
        file_path=str(source), file_offset=0, disk_pager=wrapper,
    )

    assert wrapper.prefetch([bank], [0, 1]) == 2
    assert bytes(bank.tensor[0, :4].tolist()) == b"AAAA"
    assert bytes(bank.tensor[1, :4].tolist()) == b"BBBB"
    assert wrapper.prefetch([bank], [2]) == 1
    stats = wrapper.stats()
    assert stats["fills"] == 3
    assert stats["fills_from_prefetch"] == 3
    assert stats["evictions"] == 1
    assert stats["resident_bytes"] == 2 * 4096
    assert wrapper.is_resident(bank, 2)

    victim = 0 if not wrapper.is_resident(bank, 0) else 1
    expected = ord("A") if victim == 0 else ord("B")
    assert int(bank.tensor[victim, 0]) == expected
    stats = wrapper.stats()
    assert stats["fills"] == 4
    assert stats["fault_driven"] == 1
    assert stats["evictions"] == 2
    assert sum(stats["fill_latency_histogram"]["counts"]) == 4
