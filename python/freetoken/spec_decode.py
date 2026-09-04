"""GPU-free greedy speculative decoding rules."""

from __future__ import annotations

from dataclasses import dataclass

import torch


MTP_DRAFT_STEPS = 1


@dataclass
class MTPVerifyHostStaging:
    contexts: torch.Tensor
    input_ids: torch.Tensor
    draft_token: torch.Tensor
    copy_done: torch.cuda.Event | None

    @classmethod
    def init(cls, width: int, context_len: int) -> MTPVerifyHostStaging:
        pin = torch.cuda.is_available()
        return cls(
            contexts=torch.empty(
                (width, context_len), dtype=torch.int64, pin_memory=pin
            ),
            input_ids=torch.empty(width, dtype=torch.int64, pin_memory=pin),
            draft_token=torch.empty(1, dtype=torch.int32, pin_memory=pin),
            copy_done=torch.cuda.Event() if pin else None,
        )

    def prepare(self, batch, boundary_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = batch.input_ids.numel()
        if width != self.input_ids.numel() or len(batch.padded_reqs) != 1:
            raise ValueError(
                f"MTP verify PLE staging expected one request and width "
                f"{self.input_ids.numel()}, got {len(batch.padded_reqs)} and {width}"
            )

        self.draft_token.copy_(batch.input_ids[-1:], non_blocking=True)
        if self.copy_done is not None and batch.input_ids.device.type == "cuda":
            self.copy_done.record(torch.cuda.current_stream(batch.input_ids.device))
            self.copy_done.synchronize()

        req = batch.padded_reqs[0]
        cached_len = int(req.cached_len)
        history = req.input_ids
        if history.numel() < cached_len:
            raise RuntimeError(
                f"request {req.uid} host history ends before cached_len={cached_len}"
            )
        if cached_len < history.numel():
            self.input_ids[:1].copy_(history[cached_len : cached_len + 1])
        else:
            token = req.pending_token_cpu
            done = req.sample_copy_done
            if token is None or done is None:
                if req.uid != -1 or not history.numel():
                    raise RuntimeError(
                        f"decode token for request {req.uid} is not available on the host"
                    )
                self.input_ids[:1].copy_(history[-1:])
            else:
                done.synchronize()
                self.input_ids[0].copy_(token)
        self.input_ids[1:].copy_(self.draft_token)

        context_len = self.contexts.shape[1]
        self.contexts.fill_(boundary_token_id)
        prior_len = min(context_len, cached_len)
        if prior_len:
            self.contexts[0, context_len - prior_len :].copy_(
                history[cached_len - prior_len : cached_len]
            )
        if context_len:
            self.contexts[1:, :-1].copy_(self.contexts[:-1, 1:])
            self.contexts[1:, -1].copy_(self.input_ids[:-1])
        return self.contexts, self.input_ids


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


def request_state_views(pool, kv_cache, req) -> dict:
    """Describe all mutable request-local linear and QSA state without copying it."""
    slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
    state = {"slot": slot}
    if pool is not None:
        state["conv"] = pool.conv_states[:, slot]
        state["recurrent"] = pool.recurrent_states[:, slot]
        state["slot_states"] = {
            name: value[:, slot] for name, value in pool.slot_states.items()
        }
    pending = getattr(kv_cache, "_pending_ring", None)
    if pending is not None:
        state["qsa_pending"] = pending[req.table_idx]
    return state


def snapshot_verify_state(pool, kv_cache, req) -> dict:
    """Copy all mutable request-local linear and QSA state."""
    views = request_state_views(pool, kv_cache, req)
    snapshot = {"slot": views["slot"]}
    for name in ("conv", "recurrent", "qsa_pending"):
        if name in views:
            snapshot[name] = views[name].clone()
    if "slot_states" in views:
        snapshot["slot_states"] = {
            name: value.clone() for name, value in views["slot_states"].items()
        }
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


def validate_mtp_verify_graph(value: str) -> str:
    if value not in ("off", "on"):
        raise ValueError(
            "--mtp-verify-graph must be 'off' or 'on', got " f"{value!r}"
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
    "request_state_views",
    "restore_verify_state",
    "snapshot_verify_state",
    "validate_mtp_draft_tokens",
    "validate_mtp_verify_graph",
    "validate_speculative_mtp",
]
