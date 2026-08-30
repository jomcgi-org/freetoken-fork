from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams, select_lm_head_rows
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


@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.1])
def test_temperature_zero_is_greedy_regardless_of_top_p(top_p):
    assert SamplingParams(temperature=0.0, top_p=top_p).is_greedy
    assert SamplingParams(temperature=-1.0, top_p=top_p).is_greedy


def test_positive_temperature_keeps_existing_greedy_gate():
    assert SamplingParams(temperature=0.7, top_k=1, top_p=1.0).is_greedy
    assert not SamplingParams(temperature=0.7, top_k=-1, top_p=1.0).is_greedy
    assert not SamplingParams(temperature=0.7, top_k=1, top_p=0.9).is_greedy


def test_mtp_multi_position_lm_head_row_selection():
    """Verify and one-row draft projections must bypass prefill's last-row gather."""
    width, hidden_size = 4, 3
    last = torch.tensor([width - 1])
    batch = SimpleNamespace(
        is_prefill=True,
        size=1,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]),
    )
    verify_hidden = torch.arange(width * hidden_size).view(width, hidden_size)

    # Ordinary prefill projects only the final row. MTP verification projects all
    # positions so greedy_accept_prefix receives draft_steps + 1 predictions.
    assert torch.equal(select_lm_head_rows(verify_hidden, batch), verify_hidden[-1:])
    assert select_lm_head_rows(
        verify_hidden, batch, select_last=False
    ).shape == (width, hidden_size)

    # The surrounding verify metadata still says its last row is width - 1, but
    # each autoregressive MTP draft call owns only one hidden row. Applying that
    # request-level index here was the CUDA IndexKernel out-of-bounds crash.
    draft_hidden = verify_hidden[:1]
    with pytest.raises(IndexError):
        select_lm_head_rows(draft_hidden, batch)
    assert select_lm_head_rows(
        draft_hidden, batch, select_last=False
    ).data_ptr() == draft_hidden.data_ptr()


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
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
    assert sampling_params.is_greedy
    wanted = 23

    greedy = [3]
    while len(greedy) < wanted:
        token = torch.tensor([greedy[-1]], device=device)
        greedy.append(int(model(token).argmax(dim=-1)[0]))

    speculative = [3]
    while len(speculative) < wanted:
        assert sampling_params.is_greedy  # the scheduler's MTP engagement gate
        seed = torch.tensor(speculative[-1], device=device)
        # A deliberately imperfect synthetic MTP head. The verifier must recover
        # the exact target sequence regardless of where its draft first differs.
        drafts = torch.stack(((seed + 1) % 17, (seed + 4) % 17, (seed + 5) % 17))
        verify_input = torch.cat((seed.view(1), drafts))
        targets = model(verify_input).argmax(dim=-1)
        accepted, _ = greedy_accept_prefix(drafts, targets)
        speculative.extend(accepted.tolist())

    assert speculative[:wanted] == greedy
