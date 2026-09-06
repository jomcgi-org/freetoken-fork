"""Linux file-bank checks for optional skipping of redundant populate reads."""

import os
import sys

import pytest
import torch

from freetoken.moe import host_banks, resident_range


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux file residency')


def make_bank(tmp_path):
    data = bytes(range(256)) * 64
    source = tmp_path / 'weights.ftw'
    source.write_bytes(b'abc' + data)
    bank = host_banks.HostBank((4, 4096), torch.uint8, backing='file',
                              file_path=str(source), file_offset=3)
    return bank, data


def track_reads(monkeypatch):
    original = os.preadv
    calls = []

    def read(fd, buffers, offset):
        calls.append((offset, len(buffers[0])))
        return original(fd, buffers, offset)

    monkeypatch.setattr(os, 'preadv', read)
    return calls


def test_default_path_never_probes_residency(tmp_path, monkeypatch):
    bank, data = make_bank(tmp_path)
    monkeypatch.delenv('FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT', raising=False)
    monkeypatch.setattr(resident_range, 'owned_file_probe', lambda *a: pytest.fail('default must not probe'))
    calls = track_reads(monkeypatch)
    assert bank.populate_rows([2, 0, 2], bytearray(4096)) == 8192
    assert calls == [(3, 4096), (8195, 4096)]
    assert bank.tensor.numpy().tobytes() == data


def test_warm_owned_file_skips_reads_and_preserves_mapped_bytes(tmp_path, monkeypatch):
    bank, data = make_bank(tmp_path)
    monkeypatch.setenv('FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT', '1')
    # Read the mapping once to establish resident pages for the real mincore call.
    assert bank.tensor.numpy().tobytes() == data
    calls = track_reads(monkeypatch)
    assert bank.populate_rows([2, 0, 2], bytearray(4096)) == 0
    assert calls == []
    assert bank.tensor.numpy().tobytes() == data


def test_mixed_hints_skip_only_the_reported_resident_chunks(tmp_path, monkeypatch):
    bank, data = make_bank(tmp_path)
    monkeypatch.setenv('FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT', '1')
    checked = []

    class Probe:
        def resident(self, address, length):
            checked.append((address, length))
            return len(checked) == 1

    monkeypatch.setattr(resident_range, 'owned_file_probe', lambda *a: Probe())
    calls = track_reads(monkeypatch)
    assert bank.populate_rows([0, 1, 2], bytearray(4096)) == 8192
    assert checked == [(bank.addr + n * 4096, 4096) for n in (0, 1, 2)]
    assert calls == [(4099, 4096), (8195, 4096)]
    assert bank.tensor.numpy().tobytes() == data


def test_unavailable_probe_retains_original_reads(tmp_path, monkeypatch):
    bank, data = make_bank(tmp_path)
    monkeypatch.setenv('FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT', '1')
    monkeypatch.setattr(resident_range, 'owned_file_probe', lambda *a: None)
    calls = track_reads(monkeypatch)
    assert bank.populate_rows([0, 1], bytearray(4096)) == 8192
    assert calls == [(3, 4096), (4099, 4096)]
    assert bank.tensor.numpy().tobytes() == data
