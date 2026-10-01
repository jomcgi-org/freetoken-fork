"""GPU-free checks for the decode WILLNEED critical-path helpers (issue #13)."""

import random

import pytest

from freetoken.moe import host_banks
from freetoken.moe.host_banks import coalesced_page_ranges


def _reference(expert_ids, expert_stride, *, limit=None, page_size=4096, page_offset=0):
    """The original per-page set implementation, kept as the oracle."""
    pages = set()
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
    out = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        out.append((start * page_size, (prev - start + 1) * page_size))
        start = prev = page
    out.append((start * page_size, (prev - start + 1) * page_size))
    return out


@pytest.mark.parametrize("stride", [1, 100, 4096, 5000, 12288, 70000, 1 << 20])
@pytest.mark.parametrize("page_offset", [0, 17, 4095, 8192])
def test_coalesced_page_ranges_matches_per_page_reference(stride, page_offset):
    rng = random.Random(stride * 31 + page_offset)
    for _ in range(40):
        n = rng.randint(0, 12)
        ids = [rng.choice([-1, rng.randrange(64)]) for _ in range(n)]
        ids += ids[:2]  # duplicates
        limit = 64 * stride
        assert coalesced_page_ranges(
            ids, stride, limit=limit, page_offset=page_offset
        ) == _reference(ids, stride, limit=limit, page_offset=page_offset)


def test_coalesced_page_ranges_still_rejects_out_of_range():
    with pytest.raises(ValueError):
        coalesced_page_ranges([3, 64], 4096, limit=64 * 4096)


def test_madvise_reuses_one_libc_handle(monkeypatch):
    created = []

    class FakeLibc:
        def madvise(self, *_args):
            return 0

    def fake_cdll(name, use_errno=False):
        created.append((name, use_errno))
        return FakeLibc()

    monkeypatch.setattr(host_banks, "_libc", None)
    monkeypatch.setattr(host_banks.ctypes, "CDLL", fake_cdll)
    for _ in range(5):
        host_banks._madvise(0x1000, 4096, 3)
    assert created == [(None, True)]
