"""GPU-free greedy speculative decoding rules."""

from __future__ import annotations

import torch


MTP_DRAFT_STEPS = 1


def reserve_mtp_window(batch, width: int) -> None:
    """Reserve candidate positions without changing the decode classification."""
    req = batch.reqs[0]
    batch.mtp_verify = True
    batch.mtp_original_device_len = req.device_len
    batch.mtp_original_cached_len = req.cached_len
    batch.mtp_allocated_end = req.device_len + width - 1
    req.device_len = batch.mtp_allocated_end


def configure_mtp_decode_step(
    batch,
    verify_ids: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor | None,
    step: int,
) -> None:
    """Expose one reserved candidate position as a width-1 decode operation."""
    req = batch.reqs[0]
    req.cached_len = int(batch.mtp_original_cached_len) + step
    req.device_len = int(batch.mtp_original_device_len) + step
    batch.input_ids = verify_ids[step : step + 1]
    batch.positions = positions[step : step + 1]
    if out_loc is not None:
        batch.out_loc = out_loc[step : step + 1]
    batch.phase = "decode"
    batch.mtp_fused = False


def configure_mtp_fused_step(
    batch,
    verify_ids: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor | None,
) -> None:
    """Expose ``[seed, draft]`` as one decode-routed target forward.

    The request still has one logical sequence with an extend length of two. Model
    components with recurrent state use ``mtp_fused`` to execute the two rows in
    order, while MoE dispatch continues to see a decode batch.
    """
    if verify_ids.numel() != 2 or positions.numel() != 2:
        raise ValueError("K=1 fused MTP verification requires exactly two positions")
    if out_loc is not None and out_loc.numel() != 2:
        raise ValueError("K=1 fused MTP verification requires exactly two cache locations")
    req = batch.reqs[0]
    req.cached_len = int(batch.mtp_original_cached_len)
    req.device_len = int(batch.mtp_original_device_len) + 1
    batch.input_ids = verify_ids
    batch.positions = positions
    if out_loc is not None:
        batch.out_loc = out_loc
    batch.phase = "decode"
    batch.mtp_fused = True


def snapshot_verify_state(pool, kv_cache, req) -> dict:
    """Copy all mutable request-local linear and QSA state."""
    slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
    snapshot = {"slot": slot}
    if pool is not None:
        snapshot["conv"] = pool.conv_states[:, slot].clone()
        snapshot["recurrent"] = pool.recurrent_states[:, slot].clone()
        snapshot["slot_states"] = {
            name: value[:, slot].clone() for name, value in pool.slot_states.items()
        }
    pending = getattr(kv_cache, "_pending_ring", None)
    if pending is not None:
        snapshot["qsa_pending"] = pending[req.table_idx].clone()
    return snapshot


def restore_verify_state(pool, kv_cache, req, snapshot: dict) -> None:
    """Restore a snapshot produced by :func:`snapshot_verify_state`."""
    slot = int(snapshot["slot"])
    if pool is not None and "conv" in snapshot:
        pool.conv_states[:, slot].copy_(snapshot["conv"])
        pool.recurrent_states[:, slot].copy_(snapshot["recurrent"])
        for name, value in snapshot["slot_states"].items():
            pool.slot_states[name][:, slot].copy_(value)
    pending = getattr(kv_cache, "_pending_ring", None)
    if pending is not None and "qsa_pending" in snapshot:
        pending[req.table_idx].copy_(snapshot["qsa_pending"])


def validate_speculative_mtp(value: str) -> str:
    if value not in ("off", "on"):
        raise ValueError(
            "--speculative-mtp must be 'off' or 'on', got " f"{value!r}"
        )
    return value


def validate_mtp_draft_tokens(value: int) -> int:
    if value != MTP_DRAFT_STEPS:
        raise ValueError(
            "--mtp-draft-tokens is fixed at 1 for fused MTP verification, got "
            f"{value!r}"
        )
    return value


def greedy_accept_prefix(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Accept the longest equal draft prefix and one target bonus token."""
    drafts = draft_tokens.reshape(-1)
    targets = target_tokens.reshape(-1)
    if targets.numel() < drafts.numel() + 1:
        raise ValueError(
            "target verification needs len(drafts) + 1 predictions, got "
            f"{targets.numel()} for {drafts.numel()} drafts"
        )
    matches = 0
    if drafts.numel():
        unequal = torch.nonzero(drafts != targets[: drafts.numel()])
        matches = int(unequal[0].item()) if unequal.numel() else drafts.numel()
    accepted = torch.cat((drafts[:matches], targets[matches : matches + 1]))
    return accepted, matches


__all__ = [
    "MTP_DRAFT_STEPS",
    "configure_mtp_decode_step",
    "configure_mtp_fused_step",
    "greedy_accept_prefix",
    "reserve_mtp_window",
    "restore_verify_state",
    "snapshot_verify_state",
    "validate_mtp_draft_tokens",
    "validate_speculative_mtp",
]
