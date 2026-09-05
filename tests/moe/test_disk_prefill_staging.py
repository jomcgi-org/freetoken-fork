"""Exact file transport, including ring reuse before asynchronous copies finish."""

import ctypes
import fcntl
import os

import pytest
import torch

from freetoken.moe.disk_prefill_staging import DiskPrefillStaging
from freetoken.moe.host_banks import HostBank


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("rows", [None, [7, 2, 3, 2, -1], []])
@pytest.mark.parametrize("direct_io", [False, True])
def test_staging_preserves_bank_bytes_and_unselected_rows(tmp_path, dtype, rows, direct_io):
    shape = (8, 64, 64) if direct_io else (8, 16, 32)
    nbytes = torch.empty(shape, dtype=dtype).numel() * torch.empty((), dtype=dtype).element_size()
    payload = bytes((i * 17 + i // 97) % 256 for i in range(nbytes))
    path = tmp_path / "weights.ftw"
    # An unaligned file offset catches accidental reads from the mmap base.
    path.write_bytes(b"prefix" + payload)
    bank = HostBank(shape, dtype, backing="file", file_path=str(path), file_offset=6)
    device = torch.device("cuda", torch.cuda.current_device())
    chunk_bytes = 8192 if direct_io else 97
    staging = DiskPrefillStaging(device, chunk_bytes=chunk_bytes, direct_io=direct_io)
    pointers = [buffer.data_ptr() for buffer in staging.buffers]
    try:
        for _ in range(3):
            target = torch.empty(shape, dtype=dtype, device=device)
            target.view(torch.uint8).fill_(211)
            expected = torch.full_like(bank.tensor.view(torch.uint8), 211)
            selected = list(range(8)) if rows is None else [2, 3, 7] if rows else []
            expected[selected] = bank.tensor.view(torch.uint8)[selected]
            # Delay consumption so ring reuse must retain the DMA completion fence.
            torch.cuda._sleep(1_000_000)
            copied = staging.copy_bank(bank.tensor, target, rows)
            # Consume on the same stream immediately, before synchronizing the ring.
            actual = target.view(torch.uint8).clone().cpu()
            assert torch.equal(actual, expected)
            assert copied == len(selected) * nbytes // 8
            assert [buffer.data_ptr() for buffer in staging.buffers] == pointers
            assert staging.pinned_bytes == 2 * chunk_bytes
        with pytest.raises(ValueError, match="exceeds bank size"):
            staging.copy_bank(bank.tensor, target, [8])
    finally:
        staging.synchronize()


def test_staging_retries_short_reads_and_rejects_eof(tmp_path, monkeypatch):
    path = tmp_path / "weights.ftw"
    path.write_bytes(bytes(range(256)) * 8)
    bank = HostBank((8, 256), torch.uint8, backing="file", file_path=str(path))
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=127)
    target = torch.zeros_like(bank.tensor, device=device)
    original = os.preadv

    def short_read(fd, buffers, offset):
        return original(fd, [buffers[0][:13]], offset)

    monkeypatch.setattr(os, "preadv", short_read)
    try:
        assert staging.copy_bank(bank.tensor, target) == bank.nbytes
        assert torch.equal(target.cpu(), bank.tensor)
        staging.synchronize()
        target.zero_()
        monkeypatch.setattr(os, "preadv", lambda *_args: 0)
        with pytest.raises(OSError, match="short file read"):
            staging.copy_bank(bank.tensor, target)
        assert torch.count_nonzero(target).item() == 0
    finally:
        staging.synchronize()


def test_direct_staging_aligns_shifted_buffers_and_source_views(tmp_path, monkeypatch):
    import freetoken.moe.disk_prefill_staging as module

    allocate = module.alloc_pinned_tensor

    def shifted(size, *, dtype):
        # cudaHostAlloc need not return a page-aligned pointer.
        raw = allocate(size + 4096, dtype=dtype)
        offset = (-raw.data_ptr()) % 4096 + 1
        return raw[offset:offset + size]

    monkeypatch.setattr(module, "alloc_pinned_tensor", shifted)
    payload = bytes((i * 13 + i // 71) % 256 for i in range(8 * 6000))
    path = tmp_path / "unaligned.ftw"
    path.write_bytes(b"prefix" + payload)
    bank = HostBank((8, 6000), torch.uint8, backing="file", file_path=str(path), file_offset=6)
    source = bank.tensor[1:7]
    source._freetoken_host_bank = bank
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=8192, direct_io=True)
    original = os.preadv
    reads = []

    def aligned_read(fd, buffers, offset):
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT
        assert offset % 4096 == 0
        assert len(buffers[0]) % 4096 == 0
        assert ctypes.addressof(ctypes.c_char.from_buffer(buffers[0])) % 4096 == 0
        reads.append((offset, len(buffers[0])))
        return original(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", aligned_read)
    try:
        assert staging.pinned_bytes == 16384
        assert all(buffer.numel() == 4096 for buffer in staging.buffers)
        target = torch.full_like(source, 211, device=device)
        expected = torch.full_like(source, 211)
        reference = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(8, 6000)[1:7]
        expected[[0, 2, 5]] = reference[[0, 2, 5]]
        assert staging.copy_bank(source, target, [5, 0, 2]) == 18000
        assert reads
        assert torch.equal(target.cpu(), expected)
    finally:
        staging.synchronize()


def test_direct_staging_retries_aligned_short_reads_and_rejects_eof(tmp_path, monkeypatch):
    path = tmp_path / "short.ftw"
    path.write_bytes(bytes(range(256)) * 128)
    bank = HostBank((8, 4096), torch.uint8, backing="file", file_path=str(path))
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=16384, direct_io=True)
    target = torch.zeros_like(bank.tensor, device=device)
    original = os.preadv
    calls = []

    def short_read(fd, buffers, offset):
        calls.append(offset)
        return original(fd, [buffers[0][:4096]], offset)

    monkeypatch.setattr(os, "preadv", short_read)
    try:
        assert staging.copy_bank(bank.tensor, target) == bank.nbytes
        assert len(calls) == 8
        assert torch.equal(target.cpu(), bank.tensor)
        staging.synchronize()
        monkeypatch.setattr(os, "preadv", lambda *_args: 0)
        with pytest.raises(OSError, match="short O_DIRECT read"):
            staging.copy_bank(bank.tensor, target)
        monkeypatch.setattr(os, "preadv", lambda *_args: 13)
        with pytest.raises(OSError, match="unaligned short O_DIRECT read"):
            staging.copy_bank(bank.tensor, target)
    finally:
        staging.synchronize()


@pytest.mark.parametrize("direct_io", [False, True])
def test_direct_staging_does_not_populate_file_cache(tmp_path, direct_io):
    payload = bytes(range(256)) * 256
    path = tmp_path / "residency.ftw"
    with path.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    bank = HostBank((16, 4096), torch.uint8, backing="file", file_path=str(path))
    libc = ctypes.CDLL(None, use_errno=True)

    def resident_pages():
        vec = (ctypes.c_ubyte * 16)()
        assert libc.mincore(ctypes.c_void_p(bank.addr), ctypes.c_size_t(bank.nbytes), vec) == 0
        return sum(value & 1 for value in vec)

    assert resident_pages() == 0
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=16384, direct_io=direct_io)
    try:
        target = torch.empty_like(bank.tensor, device=device)
        staging.copy_bank(bank.tensor, target)
        assert torch.equal(target.cpu().view(-1), torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        assert resident_pages() == (0 if direct_io else 16)
    finally:
        staging.synchronize()


@pytest.mark.parametrize("tokens", [16, 512])
@pytest.mark.parametrize("direct_io", [False, True])
def test_selected_staging_matches_full_nvfp4_gemm(tmp_path, tokens, direct_io):
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    experts, hidden, intermediate = 8, 256, 128
    generator = torch.Generator().manual_seed(4090)
    layouts = [
        ((experts, 2 * intermediate, hidden // 2), torch.uint8),
        ((experts, 2 * intermediate, hidden // 16), torch.float8_e4m3fn),
        ((experts, 2 * intermediate), torch.float16),
        ((experts, hidden, intermediate // 2), torch.uint8),
        ((experts, hidden, intermediate // 16), torch.float8_e4m3fn),
        ((experts, hidden), torch.float16),
    ]
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=32768, direct_io=direct_io)
    full, selected, sources = [], [], []
    try:
        for index, (shape, dtype) in enumerate(layouts):
            if dtype == torch.uint8:
                weights = torch.randint(0, 256, shape, generator=generator, dtype=dtype)
            else:
                weights = (torch.rand(shape, generator=generator) * 0.02 + 0.01).to(dtype)
            path = tmp_path / f"bank-{index}.ftw"
            path.write_bytes(weights.view(torch.uint8).numpy().tobytes())
            source = HostBank(shape, dtype, backing="file", file_path=str(path))
            sources.append(source)
            a = torch.empty(shape, dtype=dtype, device=device)
            b = torch.empty_like(a)
            # Unselected rows must not enter the GEMM, even as padding operands.
            b.fill_(255 if dtype == torch.uint8 else float("nan"))
            staging.copy_bank(source.tensor, a)
            staging.copy_bank(source.tensor, b, [1, 3, 6])
            full.append(a)
            selected.append(b)
        x = (torch.randn(tokens, hidden, generator=generator) / 4).to(device, torch.bfloat16)
        choices = torch.tensor([1, 3, 6], dtype=torch.int32)
        ids = choices[torch.randint(0, 3, (tokens, 2), generator=generator)].to(device)
        router = torch.rand(tokens, 2, generator=generator).to(device)
        reference = fused_experts_nvfp4(x, *full, router, ids, experts, "silu", False)
        actual = fused_experts_nvfp4(x, *selected, router, ids, experts, "silu", False)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    finally:
        staging.synchronize()
