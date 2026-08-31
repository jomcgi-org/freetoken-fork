"""Real userfaultfd and io_uring coverage, collected only on Linux."""

from __future__ import annotations

import ctypes
import sys

import pytest
import torch


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux userfaultfd test")


def test_uffd_page_fill_bitmap_lru_and_fault_path(tmp_path):
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
    assert stats["pages_installed"] == 4
    assert stats["rows_spanning_pages"] == 0
    assert sum(stats["fill_latency_histogram"]["counts"]) == 4


def _make_misaligned_bank(tmp_path, budget_bytes):
    try:
        from freetoken.kernel import _uffd_pager
    except ImportError:
        pytest.skip("UFFD pager extension is not built")

    try:
        _uffd_pager.probe()
        native = _uffd_pager.UffdPager(budget_bytes)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"kernel does not permit the UFFD/io_uring test: {exc}")

    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.uffd_pager import UFFDPager

    source = tmp_path / f"misaligned-{budget_bytes}.ftw"
    source.write_bytes(
        b"Z" * 1234 + b"A" * 2700 + b"B" * 2700 + b"C" * 2700
    )
    wrapper = object.__new__(UFFDPager)
    wrapper._native = native
    wrapper.budget_bytes = budget_bytes
    bank = HostBank(
        (3, 2700), torch.uint8, backing="uffd",
        file_path=str(source), file_offset=1234, disk_pager=wrapper,
    )
    return wrapper, bank


def test_uffd_misaligned_rows_and_ftw_offset(tmp_path):
    wrapper, bank = _make_misaligned_bank(tmp_path, 2 * 4096)

    assert wrapper.prefetch([bank], [0]) == 1
    assert torch.all(bank.tensor[0] == ord("A"))
    assert wrapper.is_resident(bank, 0)
    assert not wrapper.is_resident(bank, 1)

    assert int(bank.tensor[1, -1]) == ord("B")
    assert wrapper.is_resident(bank, 1)
    assert wrapper.is_resident(bank, 2)
    assert torch.all(bank.tensor[1] == ord("B"))
    assert torch.all(bank.tensor[2] == ord("C"))
    assert ctypes.string_at(bank.addr + bank.nbytes, 92) == b"\0" * 92

    stats = wrapper.stats()
    assert stats["fills"] == 2
    assert stats["fills_from_prefetch"] == 1
    assert stats["fault_driven"] == 1
    assert stats["resident_bytes"] == 2 * 4096
    assert stats["pages_installed"] == 2
    assert stats["rows_spanning_pages"] == 1
    assert sum(stats["fill_latency_histogram"]["counts"]) == 2


def test_uffd_spanning_row_counts_pages_separately(tmp_path):
    wrapper, bank = _make_misaligned_bank(tmp_path, 2 * 4096)

    assert wrapper.prefetch([bank], [1]) == 2
    stats = wrapper.stats()
    assert stats["fills"] == 1
    assert stats["fills_from_prefetch"] == 1
    assert stats["pages_installed"] == 2
    assert sum(stats["fill_latency_histogram"]["counts"]) == 1


def test_uffd_page_lru_with_misaligned_rows(tmp_path):
    wrapper, bank = _make_misaligned_bank(tmp_path, 4096)

    with pytest.raises(RuntimeError, match="expert pages"):
        wrapper.prefetch([bank], [1])
    assert wrapper.prefetch([bank], [0]) == 1
    assert int(bank.tensor[2, 0]) == ord("C")
    assert not wrapper.is_resident(bank, 0)
    assert not wrapper.is_resident(bank, 1)
    assert wrapper.is_resident(bank, 2)
    stats = wrapper.stats()
    assert stats["resident_bytes"] == 4096
    assert stats["evictions"] == 1

    assert int(bank.tensor[0, 0]) == ord("A")
    stats = wrapper.stats()
    assert stats["evictions"] == 2
    assert stats["pages_installed"] == 3
