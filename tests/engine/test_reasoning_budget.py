"""Hard reasoning-token budget: forcing logic, sampler mask, MTP gate. GPU-free."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler, force_tokens
from freetoken.reasoning_budget import ReasoningBudgetState, create_reasoning_budget_state

END = 7
OTHER = 3


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"</think>": [END], "<think>": [5], "</multi>": [8, 9]}.get(text, [])


def _state(budget, end=(END,), **kw):
    return ReasoningBudgetState(budget=budget, end_ids=tuple(end), **kw)


def test_forces_end_tag_at_exactly_n():
    state = _state(3)
    for _ in range(3):
        assert state.forced_token is None
        state.observe(OTHER)
    assert state.forced_token == END
    state.observe(END)
    assert state.done
    assert state.forced_token is None


def test_closing_before_n_has_no_effect():
    state = _state(5)
    state.observe(OTHER)
    state.observe(END)
    assert state.done
    for _ in range(10):
        state.observe(OTHER)
        assert state.forced_token is None


def test_multi_token_end_tag_is_forced_in_sequence():
    state = _state(2, end=(8, 9))
    state.observe(OTHER)
    state.observe(OTHER)
    assert state.forced_token == 8
    state.observe(8)
    assert state.forced_token == 9
    state.observe(9)
    assert state.done


def test_partial_end_tag_emitted_naturally_is_continued():
    state = _state(2, end=(8, 9))
    state.observe(OTHER)
    state.observe(8)  # budget spent mid-tag
    assert state.forced_token == 9


def test_content_after_reasoning_is_unconstrained():
    state = _state(1)
    state.observe(OTHER)
    state.observe(END)
    state.observe(END)
    assert state.forced_token is None


def test_budget_counts_only_after_model_opens_reasoning():
    state = create_reasoning_budget_state(
        {"tokens": 2, "end": "</think>", "start": "<think>", "open": False}, _Tokenizer()
    )
    for _ in range(5):  # answer text before any <think>
        state.observe(OTHER)
        assert state.forced_token is None
    state.observe(5)
    state.observe(OTHER)
    state.observe(OTHER)
    assert state.forced_token == END


def test_open_reasoning_counts_from_first_token():
    state = create_reasoning_budget_state(
        {"tokens": 1, "end": "</think>", "start": "<think>", "open": True}, _Tokenizer()
    )
    state.observe(OTHER)
    assert state.forced_token == END


def test_force_tokens_collapses_only_forced_rows():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 9.0, 3.0, 4.0]])
    force_tokens(logits, [(0, 1)])
    assert torch.argmax(logits[0]).item() == 1 and torch.isinf(logits[0, 0])
    assert logits[1].tolist() == [1.0, 9.0, 3.0, 4.0]
    # The forced token keeps its own logit, so logprobs/softmax stay well defined.
    assert logits[0, 1].item() == 2.0


def _req(budget, *, can_decode=True, guided=None):
    params = SamplingParams(
        reasoning_budget=None
        if budget is None
        else {"tokens": budget, "end": "</think>", "start": "<think>", "open": True}
    )
    return SimpleNamespace(
        sampling_params=params,
        can_decode=can_decode,
        guided_state=guided,
        reasoning_budget_state=None,
    )


def _sampler():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=16)
    sampler.set_guided_tokenizer(_Tokenizer())
    return sampler


def test_sampler_forces_after_budget_and_agrees_across_steps():
    sampler = _sampler()
    req = _req(2)
    other = _req(None)
    batch = SimpleNamespace(reqs=[other, req])
    emitted = []
    for _ in range(4):
        args = sampler.prepare(batch)
        logits = torch.zeros(2, 16)
        logits[:, OTHER] = 5.0
        sampler.sample(logits, args)  # greedy: argmax after forcing
        tokens = torch.argmax(logits, dim=-1)
        emitted.append(int(tokens[1]))
        assert int(tokens[0]) == OTHER  # unbudgeted row untouched
        sampler.finish_guided(batch, args, tokens)
    assert emitted == [OTHER, OTHER, END, OTHER]


def test_sampler_budget_free_batch_does_no_budget_work():
    sampler = _sampler()
    args = sampler.prepare(SimpleNamespace(reqs=[_req(None)]))
    assert not args.needs_host_tokens and not args.forced


def test_zero_budget_is_unlimited():
    sampler = _sampler()
    args = sampler.prepare(SimpleNamespace(reqs=[_req(0)]))
    assert not args.needs_host_tokens


def test_non_decoding_chunk_rows_are_skipped():
    sampler = _sampler()
    args = sampler.prepare(SimpleNamespace(reqs=[_req(1, can_decode=False)]))
    assert not args.budget_reqs


def test_active_grammar_rows_are_not_forced():
    sampler = _sampler()
    req = _req(1, guided=SimpleNamespace(active=True))
    req.sampling_params.guided_decoding = None
    sampler.prepare(SimpleNamespace(reqs=[req]))
    req.reasoning_budget_state.observe(OTHER)
    args = sampler.prepare(SimpleNamespace(reqs=[req]))
    assert args.forced == []
