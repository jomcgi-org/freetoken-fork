"""HOT file transport bytes, cancellation boundaries, and pinned-buffer reuse."""

import os
import threading

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.moe.host_banks import HostBank
from freetoken.moe.hot_adapt import HotSwap
from freetoken.moe.hot_staging_io import HotRowFileReader
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args


def file_bank(tmp_path, name, shape, dtype):
    size = torch.empty(shape, dtype=dtype).numel() * torch.empty((), dtype=dtype).element_size()
    payload = bytes((i * 17 + i // 11) % 256 for i in range(size))
    path = tmp_path / name
    path.write_bytes(b"prefix" + payload)
    bank = HostBank(shape, dtype, backing="file", file_path=str(path), file_offset=6)
    return bank, payload


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(6,), (6, 13)])
def test_file_rows_preserve_bytes_views_scalars_and_untouched_rows(tmp_path, monkeypatch, dtype, shape):
    bank, payload = file_bank(tmp_path, "rows.ftw", shape, dtype)
    source = bank.tensor[1:5]
    source._freetoken_host_bank = bank
    target = torch.empty((5, *shape[1:]), dtype=dtype)
    target.view(torch.uint8).fill_(211)
    row_bytes = len(payload) // shape[0]
    expected = torch.full((5, row_bytes), 211, dtype=torch.uint8)
    fds = set()
    preadv = os.preadv

    def read(fd, buffers, offset):
        fds.add(fd)
        # Exercise continuation after a successful short buffered read.
        return preadv(fd, [buffers[0][:3]], offset)

    monkeypatch.setattr(os, "preadv", read)
    reader = HotRowFileReader()
    try:
        for stage_row, source_row in enumerate((2, 0, 3)):
            assert reader.copy_row(source, source_row, target[stage_row])
            start = (source_row + 1) * row_bytes
            expected[stage_row] = torch.tensor(list(payload[start:start + row_bytes]), dtype=torch.uint8)
        assert torch.equal(target.view(torch.uint8).reshape(5, row_bytes), expected)
        assert len(fds) == 1
    finally:
        reader.close()
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_file_row_eof_is_an_error_and_descriptors_close(tmp_path, monkeypatch):
    bank, _ = file_bank(tmp_path, "eof.ftw", (4, 16), torch.uint8)
    reader = HotRowFileReader()
    target = torch.full((16,), 211, dtype=torch.uint8)
    opened = []

    def eof(fd, buffers, offset):
        opened.append(fd)
        return 0

    monkeypatch.setattr(os, "preadv", eof)
    try:
        with pytest.raises(OSError, match="short file read"):
            reader.copy_row(bank.tensor, 2, target)
        assert target.tolist() == [211] * 16
    finally:
        reader.close()
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize("fault", ["negative", "outside", "strided", "dtype", "source_view"])
def test_file_rows_reject_invalid_geometry_before_reading(tmp_path, monkeypatch, fault):
    bank, _ = file_bank(tmp_path, "bounds.ftw", (4, 16), torch.uint8)
    source, row = bank.tensor, 1
    target = torch.empty(16, dtype=torch.uint8)
    if fault == "negative":
        row = -1
    elif fault == "outside":
        row = 4
    elif fault == "strided":
        target = torch.empty(32, dtype=torch.uint8)[::2]
    elif fault == "dtype":
        target = torch.empty(16, dtype=torch.float32)
    else:
        source = torch.empty((4, 16), dtype=torch.uint8)
        source._freetoken_host_bank = bank

    def unexpected(*args):
        raise AssertionError("invalid geometry must not read a file")

    monkeypatch.setattr(os, "preadv", unexpected)
    reader = HotRowFileReader()
    try:
        with pytest.raises(ValueError):
            reader.copy_row(source, row, target)
    finally:
        reader.close()


def staging_cache(tmp_path, policy, *, cuda=False):
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=16, device=torch.device("cpu"),
        moe_hot_staging_io=policy, quant_format="nvfp4",
    )
    weights = [file_bank(tmp_path, f"layer{i}.ftw", (4, 16), torch.uint8)[0] for i in range(2)]
    # Include a non-file bank to exercise the original tensor-copy fallback.
    scales = [torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20 * i for i in range(2)]
    cache.bank_schema = ("weight", "scale")
    cache.bank_sources = {"weight": [b.tensor for b in weights], "scale": scales}
    cache._hot_staging_rows = 2
    cache._hot_staging = [
        torch.full((2, 16), 211, dtype=torch.uint8, pin_memory=cuda),
        torch.full((2, 2), -1, dtype=torch.float32, pin_memory=cuda),
    ]
    return cache


@pytest.mark.parametrize("policy,cancel", [
    ("mmap", "before"), ("mmap", "never"),
    ("buffered", "before"), ("buffered", "during"), ("buffered", "never"),
])
def test_staging_cancellation_keeps_complete_expert_prefix(tmp_path, monkeypatch, policy, cancel):
    cache = staging_cache(tmp_path, policy)
    stop = threading.Event()
    if cancel == "before":
        stop.set()
    swaps = (HotSwap(1, 0, 2, None), HotSwap(0, 1, 3, None))
    ready = []

    class Ready:
        def synchronize(self):
            ready.append(True)

    if cancel == "during":
        preadv = os.preadv

        def read(fd, buffers, offset):
            got = preadv(fd, buffers, offset)
            # Cancellation arrives after the weight bank, before its scales.
            stop.set()
            return got

        monkeypatch.setattr(os, "preadv", read)
    pointers = [t.data_ptr() for t in cache._hot_staging]
    copied, seconds = cache._stage_hot_rows(Ready(), swaps, stop)
    count = {"before": 0, "during": 1, "never": 2}[cancel]
    assert ready == [True]
    assert copied == {(s.layer_id, s.row) for s in swaps[:count]}
    assert seconds >= 0
    assert [t.data_ptr() for t in cache._hot_staging] == pointers
    for i, swap in enumerate(swaps[:count]):
        for bank_id, name in enumerate(cache.bank_schema):
            assert torch.equal(cache._hot_staging[bank_id][i], cache.bank_sources[name][swap.layer_id][swap.incoming_expert])
    if count < 2:
        assert torch.all(cache._hot_staging[0][count:] == 211)
        assert torch.all(cache._hot_staging[1][count:] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("policy", ["mmap", "buffered"])
def test_batched_hot_staging_waits_for_dma_before_reusing_buffers(tmp_path, policy):
    cache = staging_cache(tmp_path, policy, cuda=True)
    cache.device = torch.device("cuda", torch.cuda.current_device())
    cache._hot_adapt_copy_stream = torch.cuda.Stream(device=cache.device)
    cache._hot_slot_for_row = {0: [8, 9], 1: [10, 11]}
    cache.bank_caches = {
        "weight": torch.full((16, 16), 211, dtype=torch.uint8, device=cache.device),
        "scale": torch.full((16, 2), -1, dtype=torch.float32, device=cache.device),
    }
    torch.cuda.synchronize()
    swaps = (HotSwap(1, 0, 2, None), HotSwap(0, 1, 3, None), HotSwap(0, 0, 1, None))
    pointers = [t.data_ptr() for t in cache._hot_staging]
    with torch.cuda.stream(cache._hot_adapt_copy_stream):
        torch.cuda._sleep(2_000_000)
    copied, _ = cache._stage_hot_rows_batched(None, swaps)
    assert copied == {(s.layer_id, s.row) for s in swaps}
    assert [t.data_ptr() for t in cache._hot_staging] == pointers
    for name, target in cache.bank_caches.items():
        expected = torch.full_like(target, 211 if name == "weight" else -1, device="cpu")
        for swap in swaps:
            slot = cache._hot_slot_for_row[swap.layer_id][swap.row]
            expected[slot] = cache.bank_sources[name][swap.layer_id][swap.incoming_expert]
        assert torch.equal(target.cpu(), expected)


@pytest.mark.parametrize("policy", [None, "buffered"])
def test_hot_staging_cli_default_and_override(policy):
    argv = ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16"]
    if policy is not None:
        argv += ["--moe-hot-staging-io", policy]
    args, _ = parse_args(argv)
    assert args.moe_hot_staging_io == (policy or "mmap")


def test_hot_staging_rejects_unknown_policy():
    with pytest.raises(ValueError, match="must be 'mmap' or 'buffered'"):
        EngineConfig(
            model_path="/tmp/model", tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16, moe_hot_staging_io="direct",
        )
    with pytest.raises(ValueError, match="must be 'mmap' or 'buffered'"):
        OffloadMoeCache(
            num_layers=1, num_experts=4, cache_size=16, device=torch.device("cpu"),
            moe_hot_staging_io="direct",
        )


def test_buffered_hot_staging_rejects_unqualified_bank_layout():
    with pytest.raises(ValueError, match="requires native NVFP4"):
        OffloadMoeCache(
            num_layers=1, num_experts=4, cache_size=16, device=torch.device("cpu"),
            moe_hot_staging_io="buffered", quant_format="bf16",
        )
