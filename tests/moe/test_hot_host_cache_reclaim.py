"""Immutable file reclamation, cold-page preservation and HOT publication."""

import ctypes
import mmap
import os
import sys

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.moe.host_banks import HostBank
from freetoken.moe.hot_adapt import HotSwap
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args


def file_bank(tmp_path):
    page = mmap.PAGESIZE
    prefix = b"p" * (page + 13)
    payload = bytes(range(256)) * (8 * 2 * page // 256)
    path = tmp_path / "rows.ftw"
    with path.open("wb") as f:
        f.write(prefix + payload)
        f.flush()
        os.fsync(f.fileno())
    bank = HostBank((8, 2 * page), torch.uint8, backing="file",
                    file_path=str(path), file_offset=len(prefix))
    # Install every PTE, including cold rows, before testing selective eviction.
    expected = bank.tensor.clone()
    return bank, expected, path, prefix + payload


def residency(bank):
    if sys.platform != "linux":
        pytest.skip("requires Linux mincore and file-cache invalidation")
    count = (bank._mapping_length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vec = (ctypes.c_ubyte * count)()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mincore(ctypes.c_void_p(bank._mapping_addr), ctypes.c_size_t(bank._mapping_length), vec):
        raise OSError(ctypes.get_errno(), "mincore failed")
    return [bool(v & 1) for v in vec]


@pytest.mark.parametrize("view", [False, True])
def test_reclamation_drops_only_complete_hot_pages_and_preserves_all_bytes(tmp_path, view):
    bank, expected, path, payload = file_bank(tmp_path)
    assert all(residency(bank))
    source, rows = (bank.tensor[1:7], [0, 1, 4]) if view else (bank.tensor, [1, 2, 5])
    address = source.data_ptr()
    advised = bank.reclaim_file_rows(rows, tensor=source)
    # With a 13-byte header, the merged rows 1/2 contain pages 3/4/5;
    # row 5 contains page 11. Every boundary page also holds cold bytes.
    assert advised == 4 * mmap.PAGESIZE
    assert {i for i, present in enumerate(residency(bank)) if not present} == {3, 4, 5, 11}
    assert source.data_ptr() == address
    assert torch.equal(bank.tensor, expected)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("flag", ["_pinned", "_locked", "_uffd", "_tmpfs_backed", "_uvm_managed"])
def test_other_residency_owners_are_never_reclaimed(tmp_path, monkeypatch, flag):
    bank, _, _, _ = file_bank(tmp_path)
    monkeypatch.setattr(bank, flag, True)
    monkeypatch.setattr(os, "posix_fadvise", lambda *_: pytest.fail("must preserve other residency owners"))
    assert bank.reclaim_file_rows([1]) == 0


def test_tiny_scale_rows_retain_shared_pages(tmp_path, monkeypatch):
    bank, _, _, _ = file_bank(tmp_path)
    source = bank.tensor.flatten()[:512].view(8, 64)
    monkeypatch.setattr(os, "posix_fadvise", lambda *_: pytest.fail("no complete owned pages"))
    assert bank.reclaim_file_rows([1, 2, 3], tensor=source) == 0


@pytest.mark.parametrize("fault", ["outside", "strided", "dtype", "row"])
def test_invalid_views_are_rejected_before_advice(tmp_path, monkeypatch, fault):
    bank, _, _, _ = file_bank(tmp_path)
    source, rows = bank.tensor, [1]
    if fault == "outside":
        source = torch.empty_like(source)
    elif fault == "strided":
        source = source[:, ::2]
    elif fault == "dtype":
        source = source.view(torch.int32)
    else:
        rows = [8]
    monkeypatch.setattr(os, "posix_fadvise", lambda *_: pytest.fail("invalid view must not evict"))
    with pytest.raises(ValueError):
        bank.reclaim_file_rows(rows, tensor=source)


def test_advice_failure_closes_descriptor_and_keeps_source_readable(tmp_path, monkeypatch):
    bank, expected, _, _ = file_bank(tmp_path)
    fds = []
    def fail(fd, *_):
        fds.append(fd)
        raise OSError("injected advice failure")
    monkeypatch.setattr(os, "posix_fadvise", fail)
    with pytest.raises(OSError, match="injected"):
        bank.reclaim_file_rows([1])
    assert len(fds) == 1
    with pytest.raises(OSError):
        os.fstat(fds[0])
    assert torch.equal(bank.tensor, expected)


def cache_with_file_rows(tmp_path, policy, device):
    bank, expected, path, payload = file_bank(tmp_path)
    cache = OffloadMoeCache(num_layers=1, num_experts=8, cache_size=10,
                            device=torch.device(device), quant_format="nvfp4",
                            moe_hot_host_cache=policy)
    cache.bank_schema = ("weight", "global_scale")
    cache.bank_sources = {"weight": [bank.tensor], "global_scale": [torch.arange(8, dtype=torch.float32)]}
    cache.bank_caches = {name: torch.zeros((10, *rows[0].shape[1:]), dtype=rows[0].dtype, device=device)
                         for name, rows in cache.bank_sources.items()}
    cache._hot_staging_rows = 2
    cache._hot_staging = [torch.empty((2, *rows[0].shape[1:]), dtype=rows[0].dtype,
                                      pin_memory=device == "cuda") for rows in cache.bank_sources.values()]
    cache.hot_expert_capacity = {0: 2}
    cache._hot_slot_for_row = {0: [8, 9]}
    cache._hot_slots_device = torch.tensor([8, 9], dtype=torch.long, device=device)
    return cache, bank, expected, path, payload


@pytest.mark.parametrize("policy", ["retain", "reclaim"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reload_and_partial_publication_preserve_hot_weights(tmp_path, monkeypatch, policy, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cache, bank, expected, path, payload = cache_with_file_rows(tmp_path, policy, device)
    cache._hot_slot_owners = {0: [0, 1]}
    cache._reload_hot_slots()
    for row, expert in enumerate((0, 1)):
        assert torch.equal(cache.bank_caches['weight'][8 + row].cpu(), expected[expert])
    if policy == "reclaim":
        assert {i for i, present in enumerate(residency(bank)) if not present} == {1, 2, 3}
    else:
        assert all(residency(bank))
    # Repopulate all source pages, then acknowledge only the first incoming row.
    assert torch.equal(bank.tensor, expected)
    swaps = (HotSwap(0, 0, 2, 0), HotSwap(0, 1, 5, 1))
    cache._hot_slot_owners = {0: [None, None]}
    cache.hot_row_for_expert.fill_(-1)
    cache._hot_adapt_swaps_pending = swaps
    cache._hot_adapt_worker_installs = False
    cache._hot_adapt_tick_boundary = "decode"
    for stage_row, swap in enumerate(swaps):
        for bank_id, name in enumerate(cache.bank_schema):
            cache._hot_staging[bank_id][stage_row].copy_(cache.bank_sources[name][0][swap.incoming_expert])
    observed = []
    original = HostBank.reclaim_file_rows
    def reclaim(owner, rows, **kwargs):
        observed.extend(rows)
        assert cache._hot_slot_owners == {0: [2, 1]}
        assert cache.hot_row_for_expert[0].tolist() == [-1, 1, 0, -1, -1, -1, -1, -1]
        return original(owner, rows, **kwargs)
    monkeypatch.setattr(HostBank, "reclaim_file_rows", reclaim)
    cache._finish_hot_adaptation_swaps({(0, 0)})
    assert observed == ([2] if policy == "reclaim" else [])
    if policy == "reclaim":
        assert {i for i, present in enumerate(residency(bank)) if not present} == {5}
    else:
        assert all(residency(bank))
    for row, expert in enumerate((2, 1)):
        for name, rows in cache.bank_sources.items():
            assert torch.equal(cache.bank_caches[name][8 + row].cpu(), rows[0][expert])
    assert torch.equal(bank.tensor, expected)
    assert path.read_bytes() == payload


def test_reclamation_skips_retired_owners_and_warns_once(tmp_path, monkeypatch):
    cache, bank, _, _, _ = cache_with_file_rows(tmp_path, "reclaim", "cpu")
    cache._hot_slot_owners = {0: [None, 2]}
    rows, warnings = [], []
    def fail(owner, experts, **kwargs):
        rows.extend(experts)
        raise OSError("injected")
    monkeypatch.setattr(HostBank, "reclaim_file_rows", fail)
    from freetoken.moe import offload_cache
    monkeypatch.setattr(offload_cache.logger, "warning_rank0", warnings.append)
    swaps = (HotSwap(0, 0, 1, None), HotSwap(0, 1, 2, None))
    assert cache._reclaim_hot_host_rows(swaps) == 0
    assert cache._reclaim_hot_host_rows(swaps) == 0
    assert rows == [2, 2] and len(warnings) == 1


@pytest.mark.parametrize("policy", [None, "reclaim"])
def test_cli_default_and_override(policy):
    argv = ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16"]
    if policy:
        argv += ["--moe-hot-host-cache", policy]
    args, _ = parse_args(argv)
    assert args.moe_hot_host_cache == (policy or "retain")


@pytest.mark.parametrize("override", [dict(moe_disk_prefill="cpu"), dict(moe_disk_decode="gpufetch"),
                                      dict(moe_prefill_hot_split="off"), dict(moe_disk_pager="uffd"),
                                      dict(moe_bank_hugepages_tmpfs="/tmp/mirror")])
def test_incompatible_placement_is_rejected(override):
    config = dict(model_path="/tmp/model", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
                  moe_backend="offload", moe_disk_prefill="staged", moe_disk_decode="cpu",
                  moe_hot_host_cache="reclaim")
    config.update(override)
    with pytest.raises(ValueError, match="HOT host-cache reclamation requires"):
        EngineConfig(**config)
