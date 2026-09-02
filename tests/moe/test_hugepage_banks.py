"""GPU-free coverage for expert-bank transparent hugepage policy and reporting."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest


def test_hugepage_row_alignment_arithmetic():
    from freetoken.moe.host_banks import hugepage_row_alignment

    hugepage = 2 << 20
    assert hugepage_row_alignment(hugepage) == (0, 1)
    assert hugepage_row_alignment(hugepage // 2) == (0, 2)
    assert hugepage_row_alignment(hugepage // 2, hugepage // 2) == (1, 2)
    assert hugepage_row_alignment(3 * 4096, 2048) is None
    with pytest.raises(ValueError, match="positive"):
        hugepage_row_alignment(0)


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


def test_server_cli_exposes_hugepage_policy():
    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import parse_args

    assert EngineConfig.__dataclass_fields__["moe_bank_hugepages"].default == "auto"
    args, _ = parse_args([
        "--model", "/tmp/nonexistent-model",
        "--dtype", "bfloat16",
        "--moe-bank-hugepages", "off",
    ])
    assert args.moe_bank_hugepages == "off"


def test_startup_status_line_lists_banks_alignment_pin_result_and_delta():
    from freetoken.moe.host_banks import format_hugepage_status

    hugepage = 2 << 20
    status = {
        "mode": "auto",
        "backing": "mmap",
        "attempted": True,
        "advised": True,
        "reason": "MADV_HUGEPAGE accepted",
        "filesystem": "anonymous",
        "pin_before_kib": {
            "AnonHugePages": 2048,
            "FilePmdMapped": 0,
            "ShmemPmdMapped": 0,
        },
        "pin_after_kib": {
            "AnonHugePages": 2048,
            "FilePmdMapped": 0,
            "ShmemPmdMapped": 0,
        },
    }
    owner = SimpleNamespace(
        _hugepage_status=status,
        _mapping_addr=4 * hugepage,
        _mapping_length=hugepage,
    )

    class Tensor:
        _freetoken_host_bank = owner

        def stride(self, axis):
            assert axis == 0
            return hugepage // 2

        def element_size(self):
            return 1

        def data_ptr(self):
            return 4 * hugepage

    banks = SimpleNamespace(sources={"gate_up": [Tensor(), Tensor()]})
    line = format_hugepage_status(
        banks,
        "auto",
        {"AnonHugePages": 100, "FileHugePages": 200},
        {"AnonHugePages": 2148, "FileHugePages": 2248},
    )

    assert "MoE bank hugepages: mode=auto" in line
    assert "gate_up: L0-1 mmap/anonymous advised base=2MiB" in line
    assert "rows=0+every-2" in line
    assert "pin-thp=2048->2048KiB(retained)" in line
    assert "AnonHugePages=+2048KiB FileHugePages=+2048KiB" in line
    assert "kernel fault events" in line


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap and procfs probe")
def test_runtime_probe_tmpfs_and_regular_file(tmp_path):
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
        assert tmpfs_status["attempted"] is True
        assert regular_status["attempted"] is True
        assert isinstance(tmpfs_status["advised"], bool)
        assert isinstance(regular_status["advised"], bool)
        assert tmpfs_bank._mapping_addr % size == 0
        assert regular_bank._mapping_addr % size == 0
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
