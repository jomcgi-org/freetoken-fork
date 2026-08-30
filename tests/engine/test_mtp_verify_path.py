from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.spec_decode import (
    configure_mtp_decode_step,
    greedy_accept_decode,
    reserve_mtp_window,
    restore_verify_state,
    snapshot_verify_state,
)


class _Batch(SimpleNamespace):
    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"


def test_verify_state_snapshot_restore_round_trip_cpu():
    pool = SimpleNamespace(
        conv_states=torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4),
        recurrent_states=torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2),
        slot_states={
            "ple_conv": torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5),
            "ple_ngram": torch.arange(3 * 2, dtype=torch.int32).view(1, 3, 2),
        },
    )
    kv_cache = SimpleNamespace(
        _pending_ring=torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
    )
    req = SimpleNamespace(table_idx=1, linear_slot_idx=2)

    expected = {
        "conv": pool.conv_states[:, 2].clone(),
        "recurrent": pool.recurrent_states[:, 2].clone(),
        "ple_conv": pool.slot_states["ple_conv"][:, 2].clone(),
        "ple_ngram": pool.slot_states["ple_ngram"][:, 2].clone(),
        "qsa_pending": kv_cache._pending_ring[1].clone(),
    }
    snapshot = snapshot_verify_state(pool, kv_cache, req)

    pool.conv_states[:, 2].fill_(-1)
    pool.recurrent_states[:, 2].fill_(-2)
    pool.slot_states["ple_conv"][:, 2].fill_(-3)
    pool.slot_states["ple_ngram"][:, 2].fill_(-4)
    kv_cache._pending_ring[1].fill_(-5)
    restore_verify_state(pool, kv_cache, req, snapshot)

    assert torch.equal(pool.conv_states[:, 2], expected["conv"])
    assert torch.equal(pool.recurrent_states[:, 2], expected["recurrent"])
    assert torch.equal(pool.slot_states["ple_conv"][:, 2], expected["ple_conv"])
    assert torch.equal(pool.slot_states["ple_ngram"][:, 2], expected["ple_ngram"])
    assert torch.equal(kv_cache._pending_ring[1], expected["qsa_pending"])


def test_mtp_window_reservation_stays_decode():
    req = SimpleNamespace(cached_len=7, device_len=8)
    batch = _Batch(reqs=[req], phase="decode")

    reserve_mtp_window(batch, width=4)

    assert batch.is_decode
    assert not batch.is_prefill
    assert req.device_len == 11
    assert req.device_len - req.cached_len == 4
    assert batch.mtp_original_device_len == 8
    assert batch.mtp_original_cached_len == 7
    assert batch.mtp_allocated_end == 11


def test_each_reserved_verify_position_is_a_width_one_decode_op():
    req = SimpleNamespace(
        cached_len=7,
        device_len=11,
    )
    batch = _Batch(
        reqs=[req],
        phase="decode",
        mtp_original_cached_len=7,
        mtp_original_device_len=8,
    )
    verify_ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    positions = torch.arange(7, 11, dtype=torch.int32)
    out_loc = torch.arange(20, 24, dtype=torch.int32)

    seen = []
    for step in range(4):
        configure_mtp_decode_step(batch, verify_ids, positions, out_loc, step)
        seen.append(
            (
                batch.phase,
                req.device_len - req.cached_len,
                int(batch.input_ids[0]),
                int(batch.positions[0]),
                int(batch.out_loc[0]),
            )
        )

    assert seen == [
        ("decode", 1, 10, 7, 20),
        ("decode", 1, 11, 8, 21),
        ("decode", 1, 12, 9, 22),
        ("decode", 1, 13, 10, 23),
    ]


def test_decode_acceptance_stops_at_first_mismatch_without_rollback():
    drafts = torch.tensor([11, 12, 13], dtype=torch.int32)
    calls = []

    def target_step(step):
        calls.append(step)
        targets = (torch.tensor([11]), torch.tensor([99]))
        return targets[step], f"state-{step}"

    accepted, matched, state = greedy_accept_decode(drafts, target_step)

    assert accepted.tolist() == [11, 99]
    assert matched == 1
    assert state == "state-1"
    assert calls == [0, 1]
