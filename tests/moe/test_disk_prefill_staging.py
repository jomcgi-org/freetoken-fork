"""Exact file transport, including ring reuse before asynchronous copies finish."""

import os

import pytest
import torch

from freetoken.moe.disk_prefill_staging import DiskPrefillStaging
from freetoken.moe.host_banks import HostBank


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("rows", [None, [7, 2, 3, 2, -1], []])
def test_staging_preserves_bank_bytes_and_unselected_rows(tmp_path, dtype, rows):
    shape = (8, 16, 32)
    nbytes = 8 * 16 * 32 * torch.empty((), dtype=dtype).element_size()
    payload = bytes((i * 17 + i // 97) % 256 for i in range(nbytes))
    path = tmp_path / "weights.ftw"
    # An unaligned file offset catches accidental reads from the mmap base.
    path.write_bytes(b"prefix" + payload)
    bank = HostBank(shape, dtype, backing="file", file_path=str(path), file_offset=6)
    device = torch.device("cuda", torch.cuda.current_device())
    staging = DiskPrefillStaging(device, chunk_bytes=97)
    pointers = [buffer.data_ptr() for buffer in staging.buffers]
    try:
        for _ in range(3):
            target = torch.empty(shape, dtype=dtype, device=device)
            target.view(torch.uint8).fill_(211)
            expected = torch.full_like(bank.tensor.view(torch.uint8), 211)
            selected = list(range(8)) if rows is None else [2, 3, 7] if rows else []
            expected[selected] = bank.tensor.view(torch.uint8)[selected]
            copied = staging.copy_bank(bank.tensor, target, rows)
            # Consume on the same stream immediately, before synchronizing the ring.
            actual = target.view(torch.uint8).clone().cpu()
            assert torch.equal(actual, expected)
            assert copied == len(selected) * nbytes // 8
            assert [buffer.data_ptr() for buffer in staging.buffers] == pointers
            assert staging.pinned_bytes == 194
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


@pytest.mark.parametrize("tokens", [16, 512])
def test_selected_staging_matches_full_nvfp4_gemm(tmp_path, tokens):
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
    staging = DiskPrefillStaging(device, chunk_bytes=32768)
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
