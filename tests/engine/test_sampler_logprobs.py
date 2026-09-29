from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler


def _batch(*logprobs: int):
    return SimpleNamespace(
        reqs=[SimpleNamespace(sampling_params=SamplingParams(logprobs=n)) for n in logprobs]
    )


def test_batches_without_logprobs_requests_skip_the_work():
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.zeros(2, 4)
    assert sampler.logprobs(logits, torch.tensor([0, 1]), _batch(0, 0)) is None


def test_logprobs_are_the_log_softmax_at_the_sampled_token():
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    # Row 1 carries a padded vocab column that must not take probability mass.
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, 0.0], [0.0, 3.0, 1.0, 0.0, 50.0]])
    rows = sampler.logprobs(logits, torch.tensor([1, 1], dtype=torch.int32), _batch(3, 1))

    expected = torch.log_softmax(logits[:, :4], dim=-1)
    assert torch.allclose(rows.chosen, expected[:, 1])
    # Batch-wide k is the largest request; each request slices its own count.
    assert rows.top_ids.shape == (2, 3)
    assert rows.top_ids[0].tolist() == [0, 1, 2]
    assert rows.top_ids[1, 0].item() == 1
    assert math.isclose(rows.top_logprobs[0, 0].item(), expected[0, 0].item(), rel_tol=1e-6)
