"""Linux-only integration tests for the real PLE io_uring extension.

The extension is built and loaded in this module without CUDA or Triton. Real
io_uring execution first happens on node-4 because this test is skipped on the
development Mac.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="real io_uring execution requires Linux"
)

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = (
    _ROOT
    / "python"
    / "freetoken"
    / "kernel"
    / "csrc"
    / "ple_uring"
    / "ple_uring_ext.cpp"
)


@pytest.fixture(scope="module")
def uring_extension(tmp_path_factory):
    from torch.utils.cpp_extension import load

    build_dir = tmp_path_factory.mktemp("ple_uring_extension")
    return load(
        name=f"_ple_uring_integration_{os.getpid()}",
        sources=[str(_SOURCE)],
        extra_cflags=["-O2", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def _store(
    extension,
    path: Path,
    *,
    rows: int,
    row_bytes: int,
    row_stride: int,
    extent_base: int = 0,
):
    return extension.UringRowStore(
        paths=[str(path)],
        extent_file=[0],
        extent_base=[extent_base],
        rows_per_extent=rows,
        row_bytes=row_bytes,
        row_stride=row_stride,
        queue_depth=4,
    )


def _read(store, row_ids: list[int], *, row_bytes: int, stride: int):
    ids = (ctypes.c_int64 * len(row_ids))(*row_ids)
    destination = (ctypes.c_ubyte * (len(row_ids) * stride))()
    reads = store.read_rows(
        ctypes.addressof(ids),
        len(row_ids),
        ctypes.addressof(destination),
        stride,
    )
    return reads, bytes(destination)


def _open_fds() -> set[int]:
    result = set()
    for name in os.listdir("/proc/self/fd"):
        if not name.isdigit():
            continue
        fd = int(name)
        try:
            os.fstat(fd)
        except OSError:
            continue
        result.add(fd)
    return result


def test_real_ring_deduplicates_rows(uring_extension, tmp_path):
    row_bytes = 32
    row_stride = 64
    payload = bytes(index % 251 for index in range(4096))
    path = tmp_path / "dedup.bin"
    path.write_bytes(payload)
    store = _store(
        uring_extension,
        path,
        rows=8,
        row_bytes=row_bytes,
        row_stride=row_stride,
    )
    if store.direct_file_count() != 0:
        pytest.skip("O_DIRECT accepted on tmp_path; buffered fallback unavailable")
    assert len(store.direct_fallbacks()) == 1

    requested = [3, 3, 0, 7]
    reads, destination = _read(store, requested, row_bytes=row_bytes, stride=40)

    assert reads == 3
    for index, row_id in enumerate(requested):
        start = index * 40
        source = row_id * row_stride
        assert (
            destination[start : start + row_bytes]
            == payload[source : source + row_bytes]
        )


def test_real_ring_uses_direct_io_and_slices_unaligned_rows(uring_extension, tmp_path):
    direct_dir = Path(os.environ.get("FREETOKEN_TEST_DIRECT_DIR", tmp_path))
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix=".ple_uring_direct_", dir=direct_dir, delete=False
        )
    except OSError as exc:
        pytest.skip(f"cannot create O_DIRECT test file under {direct_dir}: {exc}")
    path = Path(handle.name)
    store = None
    payload = bytes(index % 251 for index in range(3 * 4096))
    try:
        with handle:
            handle.write(payload)
        store = _store(
            uring_extension,
            path,
            rows=3,
            row_bytes=32,
            row_stride=37,
            extent_base=4090,
        )
        if store.direct_file_count() != 1:
            pytest.skip("O_DIRECT refused: " + "; ".join(store.direct_fallbacks()))
        assert store.direct_file_count() == 1

        requested = [0, 1, 2]
        reads, destination = _read(store, requested, row_bytes=32, stride=40)

        assert reads == 3
        for index, row_id in enumerate(requested):
            start = index * 40
            source = 4090 + row_id * 37
            assert destination[start : start + 32] == payload[source : source + 32]
    finally:
        store = None
        path.unlink(missing_ok=True)


def test_real_ring_reports_short_read_after_truncate(uring_extension, tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(bytes(index % 251 for index in range(8192)))
    store = _store(
        uring_extension,
        path,
        rows=2,
        row_bytes=32,
        row_stride=4096,
    )
    os.truncate(path, 4096 + 8)

    with pytest.raises(
        RuntimeError, match=r"io_uring short read: got \d+ bytes, need 32"
    ):
        _read(store, [1], row_bytes=32, stride=32)


def test_real_ring_reports_io_error_for_closed_table_fd(uring_extension, tmp_path):
    path = tmp_path / "closed.bin"
    path.write_bytes(bytes(index % 251 for index in range(4096)))
    before = _open_fds()
    store = _store(
        uring_extension,
        path,
        rows=4,
        row_bytes=32,
        row_stride=32,
    )
    table_fds = [
        fd
        for fd in sorted(_open_fds() - before)
        if Path(f"/proc/self/fd/{fd}").resolve() == path.resolve()
    ]
    assert len(table_fds) == 1
    table_fd = table_fds[0]
    saved_fd = os.dup(table_fd)
    os.close(table_fd)
    try:
        with pytest.raises(RuntimeError, match="io_uring read: Bad file descriptor"):
            _read(store, [0], row_bytes=32, stride=32)
    finally:
        os.dup2(saved_fd, table_fd)
        os.close(saved_fd)
    del store


def test_poison_then_teardown_is_safe_in_subprocess(uring_extension, tmp_path):
    path = tmp_path / "poison.bin"
    path.write_bytes(bytes(index % 251 for index in range(4096)))
    script = r"""
import ctypes
import gc
import importlib.util
import os
import sys

import torch

module_name, extension_path, table_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location(module_name, extension_path)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

def open_fds():
    result = set()
    for name in os.listdir("/proc/self/fd"):
        if not name.isdigit():
            continue
        fd = int(name)
        try:
            os.fstat(fd)
        except OSError:
            continue
        result.add(fd)
    return result

before = open_fds()
store = extension.UringRowStore(
    paths=[table_path], extent_file=[0], extent_base=[0],
    rows_per_extent=4, row_bytes=32, row_stride=32, queue_depth=4,
)
ring_fds = []
for fd in open_fds() - before:
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        continue
    if target == "anon_inode:[io_uring]":
        ring_fds.append(fd)
assert len(ring_fds) == 1, ring_fds
os.close(ring_fds[0])

ids = (ctypes.c_int64 * 1)(0)
destination = (ctypes.c_ubyte * 32)()
try:
    store.read_rows(ctypes.addressof(ids), 1, ctypes.addressof(destination), 32)
except RuntimeError as error:
    assert "io_uring_enter" in str(error), error
else:
    raise AssertionError("closed ring fd did not fail")

try:
    store.read_rows(ctypes.addressof(ids), 1, ctypes.addressof(destination), 32)
except RuntimeError as error:
    assert "poisoned" in str(error), error
else:
    raise AssertionError("failed drain did not poison the row store")

del store
gc.collect()
print("poison teardown survived")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            uring_extension.__name__,
            uring_extension.__file__,
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "poison teardown survived" in result.stdout
    assert "drain failed; leaking" in result.stderr
