from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from freetoken.kvcache.disk_prefix_cache import (
    DiskPrefixStore,
    capture_hybrid_prefix_tensors,
    restore_hybrid_prefix_tensors,
)


def _payload(seed: int = 0) -> dict[str, torch.Tensor]:
    return {
        "qsa_kv": torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8) + seed,
        "qsa_index": torch.arange(2 * 4, dtype=torch.float32).view(2, 4) + seed,
        "conv": torch.arange(12, dtype=torch.bfloat16).view(3, 4) + seed,
        "recurrent": torch.arange(18, dtype=torch.float32).view(3, 2, 3) + seed,
        "slot_state.ple_conv": torch.arange(6, dtype=torch.bfloat16).view(2, 3) + seed,
        "slot_state.ple_ngram_ctx": torch.tensor([1, 2], dtype=torch.int32) + seed,
        "qsa_pending": torch.arange(8, dtype=torch.bfloat16).view(2, 4) + seed,
    }


def _store(path, identity="model-a", budget=1 << 20):
    return DiskPrefixStore(
        path,
        budget,
        identity=identity,
        checkpoint_fingerprint=identity,
        config_hash="config-a",
    )


def test_store_round_trip_with_synthetic_hybrid_state(tmp_path):
    ids = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    expected = _payload()
    store = _store(tmp_path)
    assert store.enqueue(ids, expected)
    store.flush()

    entry = store.lookup_longest(torch.tensor([11, 12, 13, 14, 99]), longer_than=0)
    assert entry is not None and entry.length == 4
    assert torch.equal(entry.tensors["token_ids"], ids)
    for name, tensor in expected.items():
        assert torch.equal(entry.tensors[name], tensor)
    stats = store.stats()
    assert stats["hits"] == 1
    assert stats["bytes_restored"] == entry.file_bytes
    store.close()


def test_lru_budget_evicts_oldest_entry(tmp_path):
    store = _store(tmp_path, budget=1 << 20)
    assert store.enqueue(torch.tensor([1, 2], dtype=torch.int32), _payload(1))
    assert store.enqueue(torch.tensor([3, 4], dtype=torch.int32), _payload(2))
    store.flush()
    files = sorted(tmp_path.glob("*.safetensors"))
    assert len(files) == 2
    older, newer = files
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))
    one_file_budget = max(path.stat().st_size for path in files)
    store.close()

    reopened = _store(tmp_path, budget=one_file_budget)
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert reopened.stats()["lru_evictions"] == 1
    reopened.close()


def test_fingerprint_mismatch_is_skipped_and_counted(tmp_path):
    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    writer = _store(tmp_path, identity="checkpoint-a")
    assert writer.enqueue(ids, _payload())
    writer.close()

    reader = _store(tmp_path, identity="checkpoint-b")
    assert reader.lookup_longest(ids) is None
    stats = reader.stats()
    assert stats["fingerprint_mismatches"] == 1
    assert stats["misses"] == 1
    reader.close()


def test_torn_write_is_removed_during_recovery(tmp_path):
    torn = tmp_path / "000000000004-dead.safetensors.tmp-crash"
    torn.write_bytes(b"partial")
    store = _store(tmp_path)
    assert not torn.exists()
    assert store.stats()["torn_writes"] == 1
    store.close()


def test_token_ids_are_verified_after_hash_lookup(tmp_path):
    ids = torch.tensor([7, 8, 9], dtype=torch.int32)
    store = _store(tmp_path)
    assert store.enqueue(ids, _payload())
    store.flush()
    path = next(tmp_path.glob("*.safetensors"))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    bad = {**_payload(), "token_ids": torch.tensor([7, 8, 10], dtype=torch.int32)}
    save_file(bad, str(path), metadata=metadata)

    assert store.lookup_longest(ids) is None
    assert not path.exists()
    assert store.stats()["corrupt_entries"] == 1
    store.close()


def test_bounded_writer_queue_drops_on_overflow(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class Fence:
        def synchronize(self):
            entered.set()
            assert release.wait(timeout=2)

    store = DiskPrefixStore(
        tmp_path, 1 << 20, identity="bounded", queue_size=1
    )
    assert store.enqueue(torch.tensor([1]), _payload(), ready=Fence())
    assert entered.wait(timeout=2)
    assert store.enqueue(torch.tensor([2]), _payload())
    assert not store.enqueue(torch.tensor([3]), _payload())
    assert store.stats()["write_drops"] == 1
    release.set()
    store.close()


def test_capture_and_restore_covers_qsa_gdn_and_ple_state():
    kv = SimpleNamespace(
        _kv_buffer=torch.arange(2 * 2 * 2 * 4 * 1 * 2, dtype=torch.float32).view(
            2, 2, 2, 4, 1, 2
        ),
        _cmp_k_buffer=torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3),
        _pending_ring=torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4),
        index_ratio=2,
        device=torch.device("cpu"),
    )
    pool = SimpleNamespace(
        conv_states=torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4),
        recurrent_states=torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2),
        slot_states={
            "ple_conv": torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5),
            "ple_ngram": torch.arange(3 * 2, dtype=torch.int32).view(1, 3, 2),
        },
        device=torch.device("cpu"),
    )
    locations = torch.arange(8, dtype=torch.int32)
    captured = capture_hybrid_prefix_tensors(
        kv, pool, kv_indices=locations, linear_slot=1, table_idx=1
    )
    expected_kv = captured["qsa_kv"].clone()
    expected_conv = captured["conv"].clone()
    expected_pending = captured["qsa_pending"].clone()

    kv._kv_buffer.zero_()
    kv._cmp_k_buffer.zero_()
    pool.conv_states[:, 2].zero_()
    pool.recurrent_states[:, 2].zero_()
    for tensor in pool.slot_states.values():
        tensor[:, 2].zero_()
    pending = restore_hybrid_prefix_tensors(
        kv, pool, captured, kv_indices=locations, linear_slot=2
    )

    assert torch.equal(kv._kv_buffer.flatten(2, 3), expected_kv)
    assert torch.equal(pool.conv_states[:, 2], expected_conv)
    assert torch.equal(pending, expected_pending)
