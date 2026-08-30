"""GPU-free greedy speculative decoding rules."""

from __future__ import annotations

import torch


MTP_DRAFT_STEPS = 3


def validate_speculative_mtp(value: str) -> str:
    if value not in ("off", "on"):
        raise ValueError(
            "--speculative-mtp must be 'off' or 'on', got " f"{value!r}"
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
    "greedy_accept_prefix",
    "validate_speculative_mtp",
]
