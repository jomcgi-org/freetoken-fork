from __future__ import annotations

import pytest
import torch

from freetoken.spec_decode import (
    greedy_accept_prefix,
    validate_speculative_mtp,
)


def test_mtp_config_defaults_off_and_accepts_on():
    assert validate_speculative_mtp("off") == "off"
    assert validate_speculative_mtp("on") == "on"


def test_mtp_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="--speculative-mtp.*off.*on"):
        validate_speculative_mtp("auto")


@pytest.mark.parametrize(
    ("drafts", "targets", "accepted", "matched"),
    [
        ([4, 5, 6], [4, 5, 6, 7], [4, 5, 6, 7], 3),
        ([4, 5, 6], [4, 9, 8, 7], [4, 9], 1),
        ([4, 5, 6], [9, 8, 7, 6], [9], 0),
        ([], [9], [9], 0),
    ],
)
def test_greedy_accept_prefix(drafts, targets, accepted, matched):
    got, got_matched = greedy_accept_prefix(torch.tensor(drafts), torch.tensor(targets))
    assert got.tolist() == accepted
    assert got_matched == matched


def test_greedy_accept_requires_bonus_prediction():
    with pytest.raises(ValueError, match=r"len\(drafts\) \+ 1"):
        greedy_accept_prefix(torch.tensor([1, 2]), torch.tensor([1, 2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_tiny_model_speculation_matches_greedy():
    class TinyTarget(torch.nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            transition = torch.arange(vocab_size).roll(-1)
            self.register_buffer("transition", transition)
            self.vocab_size = vocab_size

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (tokens.numel(), self.vocab_size),
                -1000.0,
                device=tokens.device,
            )
            logits.scatter_(1, self.transition[tokens].view(-1, 1), 0.0)
            return logits

    device = torch.device("cuda")
    model = TinyTarget(vocab_size=17).to(device)
    wanted = 23

    greedy = [3]
    while len(greedy) < wanted:
        token = torch.tensor([greedy[-1]], device=device)
        greedy.append(int(model(token).argmax(dim=-1)[0]))

    speculative = [3]
    while len(speculative) < wanted:
        seed = torch.tensor(speculative[-1], device=device)
        # A deliberately imperfect synthetic MTP head. The verifier must recover
        # the exact target sequence regardless of where its draft first differs.
        drafts = torch.stack(((seed + 1) % 17, (seed + 4) % 17, (seed + 5) % 17))
        verify_input = torch.cat((seed.view(1), drafts))
        targets = model(verify_input).argmax(dim=-1)
        accepted, _ = greedy_accept_prefix(drafts, targets)
        speculative.extend(accepted.tolist())

    assert speculative[:wanted] == greedy
