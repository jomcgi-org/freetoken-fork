"""Hermetic checks for advisory bitmap boundaries and conservative fallbacks."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'resident_range', Path(__file__).parents[1] / 'python/freetoken/moe/resident_range.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def probe(monkeypatch, outputs, max_bytes=8192):
    calls = []

    class Mincore:
        def __call__(self, address, length, pointer):
            calls.append((address, length))
            value = outputs.pop(0)
            if value is None:
                return -1
            ctypes.memmove(pointer, value, len(value))
            return 0

    monkeypatch.setattr(module.ctypes, 'CDLL', lambda *a, **kw: SimpleNamespace(mincore=Mincore()))
    monkeypatch.setattr(module.mmap, 'PAGESIZE', 4096)
    return module.ResidentRangeProbe(max_bytes), calls


def test_unaligned_range_includes_both_boundary_pages(monkeypatch):
    p, calls = probe(monkeypatch, [b'\x01\x01\x01'])
    assert p.resident(4099, 8192)
    assert calls == [(4096, 8195)]


@pytest.mark.parametrize(('bits', 'expected'), [(b'\x81\x03', True), (b'\x81\x02', False)])
def test_only_low_bit_defines_page_residency(monkeypatch, bits, expected):
    p, _ = probe(monkeypatch, [bits])
    assert p.resident(4096, 8192) is expected


def test_every_chunk_must_be_resident_and_probe_memory_stays_bounded(monkeypatch):
    p, calls = probe(monkeypatch, [b'\x01\x01', b'\x01\x00'])
    assert not p.resident(4096, 16384)
    assert calls == [(4096, 8192), (12288, 8192)]
    assert len(p._bits) == 3


def test_failed_probe_cannot_reuse_stale_resident_bits(monkeypatch):
    p, _ = probe(monkeypatch, [b'\x01', None])
    assert p.resident(4096, 4096)
    assert not p.resident(8192, 4096)


def test_foreign_owner_and_non_linux_skip_the_probe(monkeypatch):
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.os, 'fstat', lambda fd: SimpleNamespace(st_uid=11))
    monkeypatch.setattr(module.os, 'geteuid', lambda: 12)
    monkeypatch.setattr(module, 'ResidentRangeProbe', lambda _: pytest.fail('must not probe'))
    assert module.owned_file_probe(5, 8192) is None
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.os, 'fstat', lambda fd: pytest.fail('must not inspect descriptor'))
    assert module.owned_file_probe(5, 8192) is None


def test_missing_mincore_degrades_to_normal_population(monkeypatch):
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.os, 'fstat', lambda fd: SimpleNamespace(st_uid=11))
    monkeypatch.setattr(module.os, 'geteuid', lambda: 11)
    monkeypatch.setattr(module.ctypes, 'CDLL', lambda *a, **kw: SimpleNamespace())
    assert module.owned_file_probe(5, 8192) is None
