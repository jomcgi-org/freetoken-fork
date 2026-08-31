from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.spec_decode import (
    configure_mtp_decode_step,
    configure_mtp_fused_step,
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


def test_engine_config_rejects_more_than_one_mtp_draft():
    with pytest.raises(ValueError, match=r"--mtp-draft-tokens.*fixed at 1"):
        EngineConfig(
            model_path="unused",
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.bfloat16,
            mtp_draft_tokens=2,
        )


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

    reserve_mtp_window(batch, width=2)

    assert batch.is_decode
    assert not batch.is_prefill
    assert req.device_len == 9
    assert req.device_len - req.cached_len == 2
    assert batch.mtp_original_device_len == 8
    assert batch.mtp_original_cached_len == 7
    assert batch.mtp_allocated_end == 9


def test_fused_verify_exposes_seed_and_one_draft_as_one_decode_routed_step():
    req = SimpleNamespace(
        cached_len=7,
        device_len=9,
    )
    batch = _Batch(
        reqs=[req],
        phase="decode",
        mtp_original_cached_len=7,
        mtp_original_device_len=8,
    )
    verify_ids = torch.tensor([10, 11], dtype=torch.int32)
    positions = torch.arange(7, 9, dtype=torch.int32)
    out_loc = torch.arange(20, 22, dtype=torch.int32)

    configure_mtp_fused_step(batch, verify_ids, positions, out_loc)

    assert batch.phase == "decode"
    assert batch.mtp_fused
    assert req.cached_len == 7
    assert req.device_len == 9
    assert req.device_len - req.cached_len == 2
    assert torch.equal(batch.input_ids, verify_ids)
    assert torch.equal(batch.positions, positions)
    assert torch.equal(batch.out_loc, out_loc)


def test_reject_replay_reverts_to_one_decode_position():
    req = SimpleNamespace(cached_len=7, device_len=9)
    batch = _Batch(
        reqs=[req], phase="decode", mtp_original_cached_len=7,
        mtp_original_device_len=8, mtp_fused=True,
    )
    verify_ids = torch.tensor([10, 11], dtype=torch.int32)
    positions = torch.arange(7, 9, dtype=torch.int32)
    out_loc = torch.arange(20, 22, dtype=torch.int32)

    configure_mtp_decode_step(batch, verify_ids, positions, out_loc, 0)

    assert not batch.mtp_fused
    assert req.cached_len == 7
    assert req.device_len == 8
    assert req.device_len - req.cached_len == 1
    assert batch.input_ids.tolist() == [10]
    assert batch.positions.tolist() == [7]
    assert batch.out_loc.tolist() == [20]
