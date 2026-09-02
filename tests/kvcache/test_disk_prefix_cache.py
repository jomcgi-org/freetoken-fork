from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from freetoken.kvcache.disk_prefix_cache import (
    BLOCK_INDEX_TENSOR,
    BlockPresence,
    DiskPrefixStore,
    FORMAT,
    LazyKVRestore,
    capture_hybrid_prefix_tensors,
    make_block_index,
    priority_streaming_plan,
    restore_hybrid_prefix_tensors,
    token_chain_hash,
    validate_block_index,
)
from freetoken.moe.session_profile import SessionExpertProfile


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


def test_page_index_round_trip_enables_lazy_payload_lookup(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    payload = {**_payload(), BLOCK_INDEX_TENSOR: make_block_index(8, 2)}
    store = _store(tmp_path)
    assert store.enqueue(ids, payload)
    store.flush()

    entry = store.lookup_longest(ids)
    assert entry is not None and entry.supports_lazy_restore
    assert "qsa_kv" not in entry.tensors
    assert torch.equal(entry.block_index, torch.tensor([0, 2, 4, 6, 8]))
    assert torch.equal(validate_block_index(entry.block_index, 8), entry.block_index)
    store.close()


def test_lazy_restore_off_materializes_indexed_entry(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    writer = _store(tmp_path)
    assert writer.enqueue(ids, {**_payload(), BLOCK_INDEX_TENSOR: make_block_index(8, 2)})
    writer.close()

    reader = DiskPrefixStore(
        tmp_path, 1 << 20, identity="model-a", lazy_restore=False
    )
    entry = reader.lookup_longest(ids)
    assert entry is not None and not entry.supports_lazy_restore
    assert torch.equal(entry.tensors["qsa_kv"], _payload()["qsa_kv"])
    reader.close()


def test_version_one_entry_without_block_index_falls_back_to_eager(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    key = token_chain_hash("model-a", ids)
    path = tmp_path / f"{ids.numel():012d}-{key}.safetensors"
    save_file(
        {**_payload(), "token_ids": ids},
        str(path),
        metadata={
            "format": FORMAT,
            "version": "1",
            "identity": "model-a",
            "checkpoint_fingerprint": "model-a",
            "config_hash": "config-a",
            "token_count": str(ids.numel()),
            "token_hash": key,
            "prefill_tokens_per_s": "0",
        },
    )
    store = _store(tmp_path)
    entry = store.lookup_longest(ids)
    assert entry is not None and not entry.supports_lazy_restore
    assert torch.equal(entry.tensors["qsa_kv"], _payload()["qsa_kv"])
    store.close()


def test_priority_plan_keeps_sink_then_streams_newest_first():
    eager, streamed = priority_streaming_plan(num_blocks=10, hot_blocks=3)
    assert eager == (0, 9, 8, 7)
    assert streamed == (6, 5, 4, 3, 2, 1)
    assert sorted((*eager, *streamed)) == list(range(10))


def test_lazy_reader_installs_only_requested_pages_into_physical_mapping(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    payload = {**_payload(), BLOCK_INDEX_TENSOR: make_block_index(8, 2)}
    store = _store(tmp_path)
    assert store.enqueue(ids, payload)
    store.flush()
    entry = store.lookup_longest(ids)
    assert entry is not None and entry.supports_lazy_restore

    target = SimpleNamespace(
        _kv_buffer=torch.zeros(2, 3, 4, 2),
        device=torch.device("cpu"),
    )
    locations = torch.tensor([4, 5, 0, 1, 6, 7, 2, 3], dtype=torch.int32)
    restore = LazyKVRestore(
        target,
        entry,
        kv_indices=locations,
        already_resident_tokens=0,
        hot_blocks=1,
    )
    restore.install_eager()
    assert restore.presence.resident(0)
    assert restore.presence.resident(3)
    assert not restore.presence.resident(1)
    completed = []
    restore.set_on_complete(lambda: completed.append(True))
    restore._finish_once()
    assert completed == []
    restore.ensure_blocks([2, 1])

    logical = target._kv_buffer.flatten(2, 3).index_select(2, locations.to(torch.long))
    assert torch.equal(logical, payload["qsa_kv"])
    assert restore.complete
    assert completed == [True]
    store.close()


def test_lazy_reader_rejects_a_block_index_with_the_wrong_cache_page_size(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    store = _store(tmp_path)
    assert store.enqueue(
        ids, {**_payload(), BLOCK_INDEX_TENSOR: make_block_index(8, 2)}
    )
    store.flush()
    entry = store.lookup_longest(ids)
    assert entry is not None
    target = SimpleNamespace(
        _kv_buffer=torch.zeros(2, 3, 4, 2),
        _page_size=4,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="does not match cache page size"):
        LazyKVRestore(
            target,
            entry,
            kv_indices=torch.arange(8, dtype=torch.int32),
            already_resident_tokens=0,
            hot_blocks=1,
        )
    store.close()


def test_block_presence_never_publishes_a_torn_install():
    presence = BlockPresence(1)
    half_written = threading.Event()
    finish_write = threading.Event()
    installed = bytearray(8)

    def load(_block):
        installed[:4] = b"abcd"
        half_written.set()
        assert finish_write.wait(timeout=2)
        installed[4:] = b"efgh"

    owner = threading.Thread(target=lambda: presence.install(0, load))
    owner.start()
    assert half_written.wait(timeout=2)
    waiter_done = threading.Event()
    observed = []

    def wait_for_resident():
        presence.install(0, load)
        observed.append(bytes(installed))
        waiter_done.set()

    waiter = threading.Thread(target=wait_for_resident)
    waiter.start()
    assert not waiter_done.wait(timeout=0.05)
    finish_write.set()
    owner.join(timeout=2)
    waiter.join(timeout=2)
    assert observed == [b"abcdefgh"]
    assert presence.resident(0)


def test_store_round_trip_preserves_optional_versioned_expert_profile(tmp_path):
    ids = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    profile = SessionExpertProfile(
        ids=((3, 1), (), (7,)),
        counts=((9.0, 2.5), (), (4.0,)),
    )
    payload = {**_payload(), **profile.to_tensors()}
    store = _store(tmp_path)
    assert store.enqueue(ids, payload)
    store.flush()

    lightweight = store.lookup_profile_longest(torch.tensor([21, 22, 23, 24, 25]))
    assert lightweight is not None
    assert lightweight[0] == 4
    assert lightweight[1].ids == profile.ids

    entry = store.lookup_longest(ids)
    assert entry is not None
    assert entry.expert_profile is not None
    assert entry.expert_profile.ids == profile.ids
    store.close()


def test_store_entry_without_expert_profile_restores_as_before(tmp_path):
    ids = torch.tensor([31, 32, 33, 34], dtype=torch.int32)
    store = _store(tmp_path)
    assert store.enqueue(ids, _payload())
    store.flush()

    assert store.lookup_profile_longest(ids) is None
    entry = store.lookup_longest(ids)
    assert entry is not None
    assert entry.expert_profile is None
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


def test_lazy_lookup_pins_its_entry_across_an_lru_pass(tmp_path):
    ids = torch.arange(8, dtype=torch.int32)
    other_ids = torch.arange(20, 28, dtype=torch.int32)
    store = _store(tmp_path, budget=1 << 20)
    indexed = {**_payload(), BLOCK_INDEX_TENSOR: make_block_index(8, 2)}
    assert store.enqueue(ids, indexed)
    assert store.enqueue(other_ids, indexed)
    store.flush()

    entry = store.lookup_longest(ids, pin_lazy_path=True)
    assert entry is not None and entry.lazy_path_pinned
    other = next(path for path in tmp_path.glob("*.safetensors") if path != entry.path)
    os.utime(entry.path, ns=(1, 1))
    os.utime(other, ns=(2, 2))
    store.budget_bytes = max(entry.path.stat().st_size, other.stat().st_size)
    store._enforce_budget()

    assert entry.path.exists()
    assert not other.exists()
    store.release_entry_pin(entry)
    store.close()


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
