"""GPU-free coverage for expert-bank transparent hugepage policy and reporting."""

from __future__ import annotations

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _import_expert_banks_without_kernels(monkeypatch):
    """Import the public loader without importing Triton-backed cache kernels."""
    offload_cache = ModuleType("freetoken.moe.offload_cache")
    offload_cache._BANK_BYTES_PER_EXPERT = {}
    offload_cache._BANK_SCHEMAS = {}
    monkeypatch.setitem(
        sys.modules, "freetoken.moe.offload_cache", offload_cache,
    )
    monkeypatch.delitem(sys.modules, "freetoken.moe.expert_banks", raising=False)
    return importlib.import_module("freetoken.moe.expert_banks")


def _allow_config_import_without_triton(monkeypatch):
    """Import real slot-cache stat definitions while replacing Triton decorators."""
    def jit(fn=None, **_kwargs):
        return (lambda decorated: decorated) if fn is None else fn

    triton = ModuleType("triton")
    triton.jit = jit
    triton_language = ModuleType("triton.language")
    triton.language = triton_language
    flashlib_spec = importlib.util.find_spec("flashlib")
    assert flashlib_spec is not None and flashlib_spec.origin is not None
    source = (
        Path(flashlib_spec.origin).parent
        / "kernels"
        / "slot_cache"
        / "triton"
        / "lru_ensure.py"
    )
    spec = importlib.util.spec_from_file_location("_real_slot_cache_lru_ensure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as stats_patch:
        stats_patch.setitem(sys.modules, "triton", triton)
        stats_patch.setitem(sys.modules, "triton.language", triton_language)
        spec.loader.exec_module(module)
    kernels = ModuleType("flashlib.kernels")
    kernels.__path__ = []
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.N_STATS = module.N_STATS
    slot_cache.Stat = module.Stat
    monkeypatch.setitem(sys.modules, "flashlib.kernels", kernels)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)
    monkeypatch.delitem(sys.modules, "freetoken.moe.offload_cache", raising=False)


@pytest.mark.parametrize(
    ("mode", "supported", "enabled"),
    [
        ("auto", True, True),
        ("auto", False, False),
        ("on", True, True),
        ("off", True, False),
        ("off", False, False),
    ],
)
def test_hugepage_flag_gating(mode, supported, enabled):
    from freetoken.moe.host_banks import hugepages_enabled

    assert hugepages_enabled(mode, supported=supported) is enabled


def test_forced_hugepages_reject_unsupported_runtime():
    from freetoken.moe.host_banks import hugepages_enabled

    with pytest.raises(RuntimeError, match="requires Linux"):
        hugepages_enabled("on", supported=False)
    with pytest.raises(ValueError, match="auto.*on.*off"):
        hugepages_enabled("sometimes", supported=True)


def test_server_cli_exposes_hugepage_policy(monkeypatch):
    _allow_config_import_without_triton(monkeypatch)
    try:
        from freetoken.engine.config import EngineConfig
        from freetoken.server.args import parse_args

        assert EngineConfig.__dataclass_fields__["moe_bank_hugepages"].default == "auto"
        args, _ = parse_args([
            "--model", "/tmp/nonexistent-model",
            "--dtype", "bfloat16",
            "--moe-bank-hugepages", "auto",
            "--moe-bank-hugepages-tmpfs", "/dev/shm/freetoken",
            "--moe-bank-hugepages-tmpfs-margin-gib", "2.5",
        ])
        assert args.moe_bank_hugepages == "auto"
        assert args.moe_bank_hugepages_tmpfs == "/dev/shm/freetoken"
        assert args.moe_bank_hugepages_tmpfs_margin_gib == 2.5
        import torch

        with pytest.raises(ValueError, match="requires.*auto or on"):
            EngineConfig(
                "/tmp/nonexistent-model", SimpleNamespace(), torch.bfloat16,
                moe_bank_hugepages="off",
                moe_bank_hugepages_tmpfs="/dev/shm/freetoken",
            )
    finally:
        sys.modules.pop("freetoken.moe.offload_cache", None)


def test_uvm_range_exclusion_is_exact_and_never_hugepage_advised(monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks

    calls = []
    monkeypatch.setattr(mmap, "MADV_NOHUGEPAGE", 15, raising=False)
    monkeypatch.setattr(mmap, "MADV_HUGEPAGE", 14, raising=False)
    monkeypatch.setattr(host_banks, "_madvise", lambda address, length, advice: calls.append(
        (address, length, advice)
    ))
    with host_banks.requested_hugepages("auto"):
        bank = host_banks.HostBank(
            (4097,), torch.uint8, backing="mmap", uvm_managed=True,
        )
    assert calls == [(bank._mapping_addr, 8192, mmap.MADV_NOHUGEPAGE)]
    bank.register_uvm_range("synthetic HMM bank")

    assert bank._mapping_length == 8192
    assert bank._hugepage_status["uvm_excluded_bytes"] == 4097
    assert [advice for _address, _length, advice in calls] == [
        mmap.MADV_NOHUGEPAGE,
        mmap.MADV_NOHUGEPAGE,
    ]


def test_load_expert_banks_entry_point_joins_model_scope(monkeypatch):
    import torch

    from freetoken.moe import host_banks

    expert_banks = _import_expert_banks_without_kernels(monkeypatch)
    created = []

    def fake_load(*_args, **_kwargs):
        bank = host_banks.HostBank((17,), torch.uint8, backing="mmap")
        created.append(bank)
        return expert_banks.ExpertBanks("synthetic", {"weight": [bank.tensor]})

    monkeypatch.setattr(expert_banks, "_load_expert_banks_impl", fake_load)
    try:
        with host_banks.requested_hugepages("off") as scope:
            ple_bank = host_banks.HostBank((19,), torch.uint8, backing="mmap")
            result = expert_banks.load_expert_banks(
                "/synthetic",
                SimpleNamespace(),
                device=torch.device("cpu"),
                dtype=torch.uint8,
                dummy=True,
                hugepages="off",
                hugepages_tmpfs_margin_gib=0,
            )
        assert result.quant_format == "synthetic"
        assert set(scope.banks.values()) == {ple_bank, created[0]}
        assert scope.sources == {"weight": [created[0].tensor]}
        assert all(
            bank._hugepage_status["reason"] == "disabled"
            for bank in scope.banks.values()
        )
    finally:
        sys.modules.pop("freetoken.moe.expert_banks", None)


def test_tmpfs_capacity_arithmetic_and_refusal():
    from freetoken.moe.host_banks import (
        require_tmpfs_capacity,
        tmpfs_capacity_arithmetic,
    )

    assert tmpfs_capacity_arithmetic(100, 20, 80, 40) == (120, 120)
    assert require_tmpfs_capacity(100, 20, 80, 40) == (120, 120)
    with pytest.raises(
        RuntimeError,
        match=r"required=100 bank bytes \+ 20 margin bytes = 120.*"
              r"available=79 free bytes \+ 40 reusable mirror bytes = 119.*"
              r"stale-a\.bank.*stale-b\.bank",
    ):
        require_tmpfs_capacity(
            100, 20, 79, 40,
            stale_mirrors=("stale-a.bank", "stale-b.bank"),
        )


def test_tmpfs_mirror_uses_exact_range_and_reuses_identity(tmp_path, monkeypatch):
    from freetoken.moe import host_banks

    mount = tmp_path / "mirror"
    mount.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"xDATAy")
    stale = mount / "freetoken-old.pending.99999999.1"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(host_banks, "_mount_info", lambda _path: (
        "tmpfs", {"huge=always"},
    ))
    spec = [host_banks.TmpfsMirrorSource("gate_up#L00001", str(source), 1, 4)]

    targets, huge, first = host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), spec, margin_bytes=0, workers=8, chunk=2,
    )
    assert huge == "always"
    assert Path(targets[spec[0].key]).read_bytes() == b"DATA"
    assert not stale.exists()
    assert first[0] == 4
    assert first[4] == 0

    same_targets, same_huge, second = host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), spec, margin_bytes=0, workers=8, chunk=2,
    )
    assert same_targets == targets
    assert same_huge == huge
    assert second[4] == 4

    legacy = mount / ("freetoken-gate_up_L00001-" + "a" * 24 + ".bank")
    legacy.write_bytes(b"legacy")
    host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), spec, margin_bytes=0, workers=8, chunk=2,
    )
    assert not legacy.exists()

    old_target = Path(targets[spec[0].key])
    old_mtime = source.stat().st_mtime_ns
    source.write_bytes(b"xNEWWy")
    os.utime(source, ns=(old_mtime + 1_000_000_000, old_mtime + 1_000_000_000))
    changed_targets, _, third = host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), spec, margin_bytes=0, workers=8, chunk=2,
    )
    assert changed_targets != targets
    assert not old_target.exists()
    assert Path(changed_targets[spec[0].key]).read_bytes() == b"NEWW"
    assert third[4] == 0


def test_tmpfs_mirror_long_keys_do_not_evict_each_other(tmp_path, monkeypatch):
    from freetoken.moe import host_banks

    mount = tmp_path / "mirror"
    mount.mkdir()
    source_a = tmp_path / "source-a.bin"
    source_b = tmp_path / "source-b.bin"
    source_a.write_bytes(b"AAAA")
    source_b.write_bytes(b"BBBB")
    monkeypatch.setattr(host_banks, "_mount_info", lambda _path: (
        "tmpfs", {"huge=always"},
    ))
    common = "x" * 90
    spec_a = host_banks.TmpfsMirrorSource(common + "A", str(source_a), 0, 4)
    spec_b = host_banks.TmpfsMirrorSource(common + "B", str(source_b), 0, 4)

    targets, _, _ = host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), [spec_a, spec_b], margin_bytes=0, workers=2, chunk=2,
    )
    old_a = Path(targets[spec_a.key])
    live_b = Path(targets[spec_b.key])
    assert old_a.name != live_b.name
    assert old_a.read_bytes() == b"AAAA"
    assert live_b.read_bytes() == b"BBBB"

    old_mtime = source_a.stat().st_mtime_ns
    source_a.write_bytes(b"CCCC")
    os.utime(source_a, ns=(old_mtime + 1_000_000_000,) * 2)
    changed, _, _ = host_banks.prepare_tmpfs_bank_mirrors(
        str(mount), [spec_a], margin_bytes=0, workers=1, chunk=2,
    )

    assert not old_a.exists()
    assert Path(changed[spec_a.key]).read_bytes() == b"CCCC"
    assert live_b.read_bytes() == b"BBBB"


def test_tmpfs_mirror_copy_avoids_full_digest_and_hardlink(tmp_path, monkeypatch):
    from freetoken.moe import host_banks

    monkeypatch.setattr(
        host_banks.hashlib,
        "sha256",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected full digest")),
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"xDATAy")
    copied = tmp_path / "copied.bank"
    host_banks._copy_mirror(
        host_banks.TmpfsMirrorSource("copy", str(source), 1, 4),
        str(copied),
        chunk=2,
    )
    assert copied.read_bytes() == b"DATA"

    copied_whole = tmp_path / "copied-whole.bank"
    host_banks._copy_mirror(
        host_banks.TmpfsMirrorSource("link", str(source), 0, 6),
        str(copied_whole),
        chunk=2,
    )
    assert copied_whole.read_bytes() == source.read_bytes()
    assert os.stat(source).st_ino != os.stat(copied_whole).st_ino


def test_read_meminfo_hugepages_collects_anon_shmem_and_file(tmp_path):
    from freetoken.moe.host_banks import read_meminfo_hugepages

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 123 kB\n"
        "AnonHugePages: 10 kB\n"
        "ShmemHugePages: 20 kB\n"
        "FileHugePages: 30 kB\n",
        encoding="utf-8",
    )
    assert read_meminfo_hugepages(str(meminfo)) == {
        "AnonHugePages": 10,
        "ShmemHugePages": 20,
        "FileHugePages": 30,
    }

    meminfo.write_text(
        "AnonHugePages: 11 kB\nFileHugePages: 31 kB\n",
        encoding="utf-8",
    )
    assert read_meminfo_hugepages(str(meminfo)) == {
        "AnonHugePages": 11,
        "FileHugePages": 31,
    }


def test_tmpfs_banks_skip_populate_and_release_but_measure_selected_bytes():
    from freetoken.moe.host_banks import HostBank

    class Tensor:
        def stride(self, axis):
            assert axis == 0
            return 4096

        def element_size(self):
            return 1

    bank = object.__new__(HostBank)
    bank._disk = True
    bank._uffd = False
    bank._tmpfs_backed = True
    bank._view_offset = 0
    bank.nbytes = 8192
    bank.tensor = Tensor()
    assert bank.populate_rows([1, 1], bytearray(1)) == 0
    assert bank.release_rows([1, 1]) == 0
    assert bank.selected_rows_nbytes([1, 1]) == 4096


def test_auto_advises_anon_but_not_regular_files(tmp_path, monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks

    calls = []
    monkeypatch.setattr(mmap, "MADV_HUGEPAGE", 14, raising=False)
    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    monkeypatch.setattr(host_banks, "_madvise", lambda address, length, advice: calls.append(
        (address, length, advice)
    ))
    file_path = tmp_path / "bank.bin"
    file_path.write_bytes(bytes(4097))
    with host_banks.requested_hugepages("auto"):
        anonymous = host_banks.HostBank((4097,), torch.uint8, backing="mmap")
        regular = host_banks.HostBank(
            (4097,), torch.uint8, backing="file", file_path=str(file_path),
        )

    assert anonymous._hugepage_status["advised_bytes"] == 4097
    assert regular._hugepage_status["advised_bytes"] == 0
    assert regular._hugepage_status["reason"].startswith("off: regular filesystem")
    assert sum(advice == mmap.MADV_HUGEPAGE for _address, _length, advice in calls) == 1


def test_on_refuses_regular_file_without_selected_file_thp(tmp_path, monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks

    monkeypatch.setattr(mmap, "MADV_HUGEPAGE", 14, raising=False)
    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    file_path = tmp_path / "bank.bin"
    file_path.write_bytes(bytes(4096))
    with host_banks.requested_hugepages("on"):
        with pytest.raises(RuntimeError, match="cannot provide file THP"):
            host_banks.HostBank(
                (4096,), torch.uint8, backing="file", file_path=str(file_path),
            )


def test_on_excludes_ple_file_mapping_without_tmpfs_advice(tmp_path, monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks

    messages = []
    monkeypatch.setattr(mmap, "MADV_HUGEPAGE", 14, raising=False)
    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    monkeypatch.setattr(host_banks.logger, "info", messages.append)
    file_path = tmp_path / "ple.bin"
    file_path.write_bytes(bytes(4096))
    with host_banks.requested_hugepages("on"):
        bank = host_banks.HostBank(
            (4096,), torch.uint8, backing="file", file_path=str(file_path),
            file_thp_exclusion="PLE data shard",
        )

    assert bank._hugepage_status["reason"] == (
        "excluded: file THP not available for PLE data shard"
    )
    assert messages == [
        "MoE bank hugepages: excluded: file THP not available for PLE data shard"
    ]
    assert all("tmpfs" not in message for message in messages)


def test_tmpfs_file_mapping_skips_madv_random(tmp_path, monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks

    calls = []

    class FakeAlignedFileMapping:
        def __init__(self, _fd, length, _offset):
            self.address = 2 << 20
            self.length = length
            self.buffer = memoryview(bytearray(length)).toreadonly()

    monkeypatch.setattr(mmap, "MADV_RANDOM", 1, raising=False)
    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    monkeypatch.setattr(host_banks, "_AlignedFileMapping", FakeAlignedFileMapping)
    monkeypatch.setattr(host_banks, "_madvise", lambda *args: calls.append(args))
    monkeypatch.setattr(host_banks, "_filesystem_type", lambda _path: "tmpfs")
    file_path = tmp_path / "tmpfs.bank"
    file_path.write_bytes(bytes(4096))
    with host_banks.requested_hugepages("auto"):
        bank = host_banks.HostBank(
            (4096,), torch.uint8, backing="file", file_path=str(file_path),
            tmpfs_backed=True, tmpfs_huge="always",
        )

    assert bank._hugepage_status["reason"] == "tmpfs huge=always"
    assert calls == []


def test_tmpfs_aligned_mapping_auto_falls_back_but_on_is_strict(
    tmp_path, monkeypatch,
):
    import mmap

    import torch

    from freetoken.moe import host_banks

    monkeypatch.setattr(host_banks, "hugepages_supported", lambda **_kwargs: True)
    monkeypatch.setattr(
        host_banks,
        "_AlignedFileMapping",
        lambda *_args: (_ for _ in ()).throw(OSError("no aligned VMA")),
    )
    monkeypatch.setattr(host_banks, "_filesystem_type", lambda _path: "tmpfs")
    file_path = tmp_path / "tmpfs.bank"
    file_path.write_bytes(bytes(4096))

    with host_banks.requested_hugepages("auto"):
        bank = host_banks.HostBank(
            (4096,), torch.uint8, backing="file", file_path=str(file_path),
            tmpfs_backed=True, tmpfs_huge="always",
        )
    assert isinstance(bank._mapping, mmap.mmap)
    assert bank._hugepage_status["alignment_error"] == "no aligned VMA"
    assert "unaligned mmap fallback" in bank._hugepage_status["reason"]

    with host_banks.requested_hugepages("on"):
        with pytest.raises(OSError, match="no aligned VMA"):
            host_banks.HostBank(
                (4096,), torch.uint8, backing="file", file_path=str(file_path),
                tmpfs_backed=True, tmpfs_huge="always",
            )


def test_cuda_nohugepage_is_deferred_until_strict_uvm_registration(monkeypatch):
    import mmap

    import torch

    from freetoken.moe import host_banks
    import freetoken.kernel.pinned as pinned

    monkeypatch.setattr(mmap, "MADV_NOHUGEPAGE", 15, raising=False)
    monkeypatch.setattr(host_banks.sys, "platform", "linux")
    monkeypatch.setattr(
        host_banks, "_madvise",
        lambda *_args: (_ for _ in ()).throw(OSError("driver VMA refused")),
    )
    monkeypatch.setattr(
        pinned, "alloc_pinned_tensor",
        lambda *shape, dtype: torch.empty(shape, dtype=dtype),
    )

    def bare_bank(backing):
        bank = object.__new__(host_banks.HostBank)
        bank.nbytes = 4097
        bank._mapping_addr = 0x100000
        bank._mapping_length = 8192
        bank._view_offset = 0
        bank._hugepage_status = {
            "backing": backing,
            "attempted": False,
            "reason": "not attempted",
            "uvm_excluded_bytes": 0,
        }
        return bank

    cuda_bank = host_banks.HostBank((4097,), torch.uint8, backing="cuda")
    assert cuda_bank._uvm_managed is False
    assert cuda_bank._uvm_registered is False
    assert cuda_bank._hugepage_status["uvm_excluded_bytes"] == 0
    with pytest.raises(RuntimeError, match="cannot exclude 4097 UVM/HMM bytes"):
        cuda_bank.register_uvm_range("PLE scale shard 0")
    with pytest.raises(RuntimeError, match="cannot exclude 4097 UVM/HMM bytes"):
        bare_bank("mmap")._apply_uvm_exclusion()

    accepted = host_banks.HostBank((4097,), torch.uint8, backing="cuda")
    calls = []
    monkeypatch.setattr(
        host_banks, "_madvise",
        lambda address, length, advice: calls.append((address, length, advice)),
    )
    accepted.register_uvm_range("PLE scale shard 0")
    assert accepted._uvm_registered is True
    assert accepted._hugepage_status["uvm_excluded_bytes"] == 4097
    assert calls == [(accepted._mapping_addr, 8192, mmap.MADV_NOHUGEPAGE)]
    monkeypatch.setattr(host_banks, "_mappings_huge_kib", lambda _mappings: {})
    report = host_banks.format_hugepage_status(
        host_banks.HugepageLoadScope("off", {id(cuda_bank): cuda_bank}),
        "off", None, None,
    )
    assert "MoE bank hugepages [anonymous]: policy=off" in report
    assert "MoE bank hugepages [uvm]: policy=off thp=none; banks=0" in report


def test_uvm_exclusion_never_starts_below_allocation(monkeypatch):
    import mmap

    from freetoken.moe import host_banks

    bank = object.__new__(host_banks.HostBank)
    bank.nbytes = 2048
    bank._mapping_addr = 0x100123
    bank._mapping_length = 8192
    bank._view_offset = 512
    bank._hugepage_status = {
        "attempted": False,
        "reason": "not attempted",
        "uvm_excluded_bytes": 0,
    }
    calls = []
    monkeypatch.setattr(mmap, "MADV_NOHUGEPAGE", 15, raising=False)
    monkeypatch.setattr(
        host_banks, "_madvise",
        lambda address, length, advice: calls.append((address, length, advice)),
    )

    assert bank._apply_uvm_exclusion() == 2048
    assert calls[0][0] == bank._mapping_addr


def test_startup_status_has_class_mapping_and_single_meminfo_lines(monkeypatch):
    from freetoken.moe import host_banks

    hugepage = 2 << 20
    status = {
        "mode": "auto",
        "backing": "mmap",
        "attempted": True,
        "advised": True,
        "reason": "MADV_HUGEPAGE accepted",
        "filesystem": "anonymous",
        "pin_thp_kib": (2048, 1024),
        "advised_bytes": hugepage,
        "uvm_excluded_bytes": 0,
        "uvm_registration": None,
    }
    owner = SimpleNamespace(
        _hugepage_status=status,
        _mapping_addr=4 * hugepage,
        _mapping_length=hugepage,
        _uvm_managed=False,
        _uvm_registered=False,
        nbytes=hugepage,
    )
    uvm_status = dict(status)
    uvm_status.update(
        advised=False,
        advised_bytes=0,
        reason="MADV_NOHUGEPAGE accepted",
        uvm_excluded_bytes=4097,
        uvm_registration="PLE data shard 0",
    )
    uvm_owner = SimpleNamespace(
        _hugepage_status=uvm_status,
        _mapping_addr=6 * hugepage + 4096,
        _mapping_length=8192,
        _uvm_managed=True,
        _uvm_registered=True,
        nbytes=4097,
    )

    file_status = dict(status)
    file_status.update(
        backing="file",
        advised=True,
        reason="tmpfs huge=always",
        pin_thp_kib=None,
    )
    file_owner = SimpleNamespace(
        _hugepage_status=file_status,
        _mapping_addr=8 * hugepage,
        _mapping_length=hugepage,
        _tmpfs_backed=True,
        _uvm_registered=False,
        nbytes=hugepage,
    )

    class Tensor:
        _freetoken_host_bank = owner
        ndim = 2

        def stride(self, axis):
            assert axis == 0
            return hugepage // 2

        def element_size(self):
            return 1

        def data_ptr(self):
            return 4 * hugepage

    class UVMTensor(Tensor):
        _freetoken_host_bank = uvm_owner

        def data_ptr(self):
            return 6 * hugepage + 4096

    class FileTensor(Tensor):
        _freetoken_host_bank = file_owner

        def data_ptr(self):
            return 8 * hugepage

    owner.tensor = Tensor()
    uvm_owner.tensor = UVMTensor()
    file_owner.tensor = FileTensor()

    transient_status = dict(status)
    transient_status["reason"] = "transient staging"
    transient_status["transient"] = True
    transient_owner = SimpleNamespace(
        _hugepage_status=transient_status,
        _mapping_addr=10 * hugepage,
        _mapping_length=hugepage,
        _uvm_registered=False,
        nbytes=hugepage,
    )
    banks = host_banks.HugepageLoadScope(
        "auto",
        {
            id(owner): owner,
            id(uvm_owner): uvm_owner,
            id(file_owner): file_owner,
            id(transient_owner): transient_owner,
        },
        sources={
            "gate_up": [Tensor(), Tensor()],
            "down": [UVMTensor(), FileTensor()],
            "PLE table": [FileTensor()],
        },
    )
    observed = {
        id(owner): {
            "AnonHugePages": 1024,
            "FilePmdMapped": 0,
            "ShmemPmdMapped": 0,
        },
        id(uvm_owner): {
            "AnonHugePages": 0,
            "FilePmdMapped": 0,
            "ShmemPmdMapped": 0,
        },
        id(file_owner): {
            "AnonHugePages": 0,
            "FilePmdMapped": 0,
            "ShmemPmdMapped": 0,
        },
    }
    monkeypatch.setattr(host_banks, "_mappings_huge_kib", lambda _maps: observed)
    line = host_banks.format_hugepage_status(
        banks,
        "auto",
        {"AnonHugePages": 100, "ShmemHugePages": 300, "FileHugePages": 200},
        {"AnonHugePages": 2148, "ShmemHugePages": 2348, "FileHugePages": 2248},
    )

    lines = line.splitlines()
    assert len(lines) == 7
    assert "MoE bank hugepages [anonymous]: policy=auto" in lines[0]
    assert (
        f"banks=1 bytes_advised={hugepage} bytes_excluded_uvm=0 "
        "registered_uvm_ranges=0 registered_uvm_bytes=0"
    ) in lines[0]
    assert "MoE bank hugepages [file]" in lines[1]
    assert "tmpfs_thp=+2048KiB (ShmemHugePages meminfo delta)" in lines[1]
    assert "smaps_thp=0KiB (at load, before touch)" in lines[1]
    assert "MoE bank hugepages [uvm]" in lines[2]
    assert (
        "banks=1 bytes_advised=0 bytes_excluded_uvm=4097 "
        "registered_uvm_ranges=1 registered_uvm_bytes=4097"
    ) in lines[2]
    assert lines[3].startswith("MoE bank hugepages [gate_up]: L0-1 ")
    assert "pin-thp: kept 0/2, retained 1024 KiB" in lines[3]
    assert "base: 2MiB-aligned 2/2, rows: 2MiB-aligned 2/2" in lines[3]
    assert lines[4].startswith("MoE bank hugepages [down]: L0 ")
    assert ", L1 " in lines[4]
    assert lines[5].startswith("MoE bank hugepages [PLE table]: L0 file/")
    assert "bytes=" not in lines[5]
    assert "mapping " not in line
    assert "transient staging" not in line
    assert lines[-1] == (
        "MoE bank hugepages [meminfo delta]: "
        "AnonHugePages=+2048KiB ShmemHugePages=+2048KiB "
        "FileHugePages=+2048KiB"
    )
    assert sum("AnonHugePages=+2048KiB" in value for value in lines) == 1


def test_startup_status_groups_banks_with_differing_retained_kib(monkeypatch):
    from freetoken.moe import host_banks

    hugepage = 2 << 20
    tensors = []
    retained_values = [1920, 1936, 1952, 1968, 1984, 2000, 2016, 2048]
    for layer_id, retained_kib in enumerate(retained_values):
        owner = SimpleNamespace(
            _hugepage_status={
                "backing": "mmap",
                "filesystem": "anonymous",
                "reason": "MADV_HUGEPAGE accepted",
                "pin_thp_kib": (4096, retained_kib),
                "advised_bytes": 0,
                "uvm_excluded_bytes": 0,
                "uvm_registration": None,
            },
            _mapping_addr=(layer_id + 1) * hugepage + (layer_id + 1) * 4096,
            _mapping_length=hugepage,
            _uvm_registered=False,
            nbytes=hugepage + layer_id,
        )

        class Tensor:
            ndim = 2
            _freetoken_host_bank = owner
            _ptr = owner._mapping_addr

            def data_ptr(self):
                return self._ptr

        tensor = Tensor()
        owner.tensor = tensor
        tensors.append(tensor)

    monkeypatch.setattr(host_banks, "_mappings_huge_kib", lambda _maps: {})
    report = host_banks.format_hugepage_status(
        SimpleNamespace(sources={"gate_up": tensors}), "off", None, None,
    )
    detail = report.splitlines()[3]

    assert detail.startswith(
        "MoE bank hugepages [gate_up]: L0-7 "
        "mmap/anonymous MADV_HUGEPAGE accepted; "
    )
    assert "pin-thp: kept 0/8, retained 1920..2048 KiB" in detail
    assert "base: 2MiB-aligned 0/8, rows: 2MiB-aligned 0/8" in detail
    assert ", L" not in detail
    assert "base-offset=" not in detail
    assert "bytes=" not in detail


def test_startup_status_distinguishes_aligned_and_misaligned_rows(monkeypatch):
    from freetoken.moe import host_banks

    hugepage = 2 << 20

    def tensor_at(address):
        owner = SimpleNamespace(
            _hugepage_status={
                "backing": "mmap",
                "filesystem": "anonymous",
                "reason": "disabled",
                "pin_thp_kib": None,
                "advised_bytes": 0,
                "uvm_excluded_bytes": 0,
                "uvm_registration": None,
            },
            _mapping_addr=address,
            _mapping_length=hugepage,
            _uvm_registered=False,
            nbytes=hugepage,
        )

        class Tensor:
            ndim = 2
            _freetoken_host_bank = owner

            def data_ptr(self):
                return address

        tensor = Tensor()
        owner.tensor = tensor
        return tensor

    sources = {
        "aligned": [tensor_at(2 * hugepage), tensor_at(4 * hugepage)],
        "misaligned": [
            tensor_at(6 * hugepage + 4096),
            tensor_at(8 * hugepage + 8192),
        ],
    }
    monkeypatch.setattr(host_banks, "_mappings_huge_kib", lambda _maps: {})
    report = host_banks.format_hugepage_status(
        SimpleNamespace(sources=sources), "off", None, None,
    )
    lines = report.splitlines()

    aligned = next(line for line in lines if "[aligned]" in line)
    misaligned = next(line for line in lines if "[misaligned]" in line)
    assert "rows: 2MiB-aligned 2/2" in aligned
    assert "rows: 2MiB-aligned 0/2" in misaligned
    assert aligned != misaligned


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fixed-address tmpfs mmap")
def test_tmpfs_bank_uses_real_two_mib_aligned_mapping():
    import torch

    from freetoken.moe.host_banks import (
        HostBank,
        _AlignedFileMapping,
        hugepages_supported,
        requested_hugepages,
    )

    if not hugepages_supported():
        pytest.skip("runtime has no MADV_HUGEPAGE")
    shm = "/dev/shm"
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        pytest.skip("tmpfs /dev/shm is unavailable")
    path = os.path.join(shm, f"freetoken-aligned-test-{os.getpid()}")
    try:
        with open(path, "wb") as handle:
            handle.truncate(2 << 20)
        with requested_hugepages("auto"):
            bank = HostBank(
                (2 << 20,), torch.uint8, backing="file", file_path=path,
                tmpfs_backed=True, tmpfs_huge="always",
            )
        assert isinstance(bank._mapping, _AlignedFileMapping)
        assert bank._mapping_addr % (2 << 20) == 0
        assert bank.tensor[0].item() == 0
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap and procfs probe")
def test_unselected_file_mappings_stay_off(tmp_path):
    import torch

    from freetoken.moe.host_banks import (
        HostBank,
        hugepages_supported,
        requested_hugepages,
    )

    if not hugepages_supported():
        pytest.skip("runtime has no MADV_HUGEPAGE")
    shm = "/dev/shm"
    if not os.path.isdir(shm) or not os.access(shm, os.W_OK):
        pytest.skip("tmpfs /dev/shm is unavailable")
    tmpfs_path = os.path.join(shm, f"freetoken-thp-test-{os.getpid()}")
    regular_path = tmp_path / "regular-bank.bin"
    size = 2 << 20
    try:
        with open(tmpfs_path, "wb") as handle:
            handle.truncate(size)
        with open(regular_path, "wb") as handle:
            handle.truncate(size)
        with requested_hugepages("auto"):
            tmpfs_bank = HostBank(
                (size,), torch.uint8, backing="file", file_path=tmpfs_path
            )
            regular_bank = HostBank(
                (size,), torch.uint8, backing="file", file_path=str(regular_path)
            )

        tmpfs_status = tmpfs_bank._hugepage_status
        regular_status = regular_bank._hugepage_status
        if regular_status["filesystem"] == "tmpfs":
            pytest.skip("pytest temporary directory is also tmpfs")
        assert tmpfs_status["filesystem"] == "tmpfs"
        assert regular_status["filesystem"] != "tmpfs"
        assert tmpfs_status["attempted"] is False
        assert regular_status["attempted"] is False
        assert isinstance(tmpfs_status["advised"], bool)
        assert isinstance(regular_status["advised"], bool)
        assert tmpfs_bank.tensor[0].item() == 0
        assert regular_bank.tensor[0].item() == 0
    finally:
        try:
            os.unlink(tmpfs_path)
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux MADV_HUGEPAGE probe")
def test_anonymous_bank_mapping_is_two_mib_aligned():
    import torch

    from freetoken.moe.host_banks import (
        HostBank,
        hugepages_supported,
        requested_hugepages,
    )

    if not hugepages_supported():
        pytest.skip("runtime has no MADV_HUGEPAGE")
    with requested_hugepages("auto"):
        bank = HostBank((4097,), torch.uint8, backing="mmap")
    assert bank._mapping_addr % (2 << 20) == 0
    assert len(bank._buf) == 2 << 20
    assert bank._hugepage_status["attempted"] is True
    assert bank.tensor.numel() == 4097


def test_off_policy_preserves_page_rounded_mapping():
    import torch

    from freetoken.moe.host_banks import HostBank, requested_hugepages

    with requested_hugepages("off"):
        bank = HostBank((4097,), torch.uint8, backing="mmap")
    assert len(bank._buf) == 8192
    assert bank._hugepage_status["attempted"] is False
    assert bank._hugepage_status["reason"] == "disabled"
