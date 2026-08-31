"""GPU-free tests for the portable UFFD pager surface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeNativePager:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.regions = []
        self.prefetches = []

    def add_region(self, *args):
        self.regions.append(args)
        return len(self.regions) - 1

    def prefetch(self, regions, rows):
        self.prefetches.append((regions, rows))
        return 11

    def is_resident(self, region, row):
        return (region, row) == (0, 3)

    def stats(self, reset):
        return {
            "fills": 4,
            "fills_from_prefetch": 3,
            "fault_driven": 1,
            "evictions": 2,
            "resident_bytes": 8192,
            "pages_installed": 4,
            "rows_spanning_pages": 1,
            "fill_latency_histogram": {
                "buckets_us": [50, 100],
                "counts": [1, 2, 1],
            },
        }

    def raise_if_error(self):
        return None


class _FakeNativeModule:
    def __init__(self):
        self.probes = 0
        self.instances = []

    def probe(self):
        self.probes += 1

    def UffdPager(self, budget_bytes):
        pager = _FakeNativePager(budget_bytes)
        self.instances.append(pager)
        return pager


def test_non_linux_probe_fails_before_native_import(monkeypatch):
    import freetoken.moe.uffd_pager as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux-only"):
        module.probe_uffd_support()


def test_server_cli_exposes_uffd_pager_flags():
    from freetoken.server.args import parse_args

    args, _ = parse_args([
        "--model", "/tmp/nonexistent-model",
        "--dtype", "bfloat16",
        "--moe-disk-pager", "uffd",
        "--moe-disk-lookahead", "off",
        "--moe-pager-budget-gib", "12.5",
    ])
    assert args.moe_disk_pager == "uffd"
    assert args.moe_disk_lookahead == "off"
    assert args.moe_pager_budget_gib == 12.5


def test_probe_reports_sysctl_requirement(monkeypatch):
    import freetoken.moe.uffd_pager as module

    native = SimpleNamespace(probe=lambda: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.Path, "read_text", lambda *args, **kwargs: "0\n")
    with pytest.raises(RuntimeError, match=r"unprivileged_userfaultfd=1.*='0'"):
        module.probe_uffd_support(_native_module=native)


def test_python_wrapper_deduplicates_rows_and_exposes_stats(monkeypatch):
    import freetoken.moe.uffd_pager as module

    native = _FakeNativeModule()
    monkeypatch.setattr(module.sys, "platform", "linux")
    pager = module.UFFDPager(16384, _native_module=native)
    bank0 = SimpleNamespace(_pager_region=0)
    bank1 = SimpleNamespace(_pager_region=1)

    assert native.probes == 1
    assert pager.prefetch([bank0, bank1], [3, 1, 3, -1]) == 11
    assert native.instances[0].prefetches == [([0, 1], [1, 3])]
    assert pager.is_resident(bank0, 3)
    assert pager.stats()["fills_from_prefetch"] == 3
    assert pager.stats()["pages_installed"] == 4

    row_bank = SimpleNamespace(nbytes=32768, tensor=SimpleNamespace(shape=(8, 4096)))
    with pytest.raises(ValueError, match="working set"):
        pager.validate_working_set([row_bank], 5, context="decode")


def test_register_bank_preserves_misaligned_row_and_file_offset(monkeypatch):
    import freetoken.moe.uffd_pager as module

    native = _FakeNativeModule()
    monkeypatch.setattr(module.sys, "platform", "linux")
    pager = module.UFFDPager(8192, _native_module=native)
    bank = SimpleNamespace(addr=0x1000, _buf=bytearray(8192), nbytes=8100)

    assert pager.register_bank(
        bank,
        file_path="/tmp/misaligned.ftw",
        file_offset=1234,
        row_bytes=2700,
        num_rows=3,
    ) == 0
    assert native.instances[0].regions == [(
        0x1000,
        8192,
        8100,
        "/tmp/misaligned.ftw",
        1234,
        2700,
        3,
    )]
    logical_bank = SimpleNamespace(
        nbytes=8100,
        tensor=SimpleNamespace(shape=(3, 2700)),
    )
    pager.validate_working_set([logical_bank], 1, context="decode")
    pager.budget_bytes = 4096
    with pytest.raises(ValueError, match="expert-page working set"):
        pager.validate_working_set([logical_bank], 1, context="decode")


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_uffd_budget_validation(monkeypatch, budget):
    import freetoken.moe.uffd_pager as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(ValueError, match="finite positive"):
        module.make_uffd_pager(budget, _native_module=_FakeNativeModule())
