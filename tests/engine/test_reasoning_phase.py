"""Phase tracking for the HOT reasoning/answer histories (split3) and its flag. GPU-free."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.sample import Sampler
from freetoken.reasoning_budget import ReasoningBudgetState, create_reasoning_budget_state
from freetoken.server.generation import GenSpec, resolve_reasoning_phase

START, END, OTHER = 5, 7, 3


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"</think>": [END], "<think>": [START]}.get(text, [])


def _req(spec):
    return SimpleNamespace(
        sampling_params=SamplingParams(reasoning_phase=spec),
        reasoning_phase_state=None,
    )


def _sampler():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=16)
    sampler.set_guided_tokenizer(_Tokenizer())
    return sampler


def test_zero_budget_tracker_never_forces_and_follows_tags():
    state = ReasoningBudgetState(budget=0, end_ids=(END,), start_ids=(START,), in_reasoning=False)
    assert not state.reasoning_active and state.forced_token is None
    state.observe(OTHER)
    assert not state.reasoning_active
    state.observe(START)
    assert state.reasoning_active
    for _ in range(50):
        state.observe(OTHER)
        assert state.forced_token is None
    state.observe(END)
    assert not state.reasoning_active


def test_open_template_starts_in_reasoning_until_end_tag():
    state = create_reasoning_budget_state(
        {"end": "</think>", "start": "<think>", "open": True}, _Tokenizer()
    )
    assert state.budget == 0 and state.reasoning_active
    state.observe(OTHER)
    assert state.reasoning_active
    state.observe(END)
    assert not state.reasoning_active


def test_batch_in_reasoning_is_any_row():
    sampler = _sampler()
    thinking = _req({"end": "</think>", "start": "<think>", "open": True})
    chat = _req({"end": "</think>", "start": "<think>", "open": False})
    untracked = _req(None)

    assert sampler.batch_in_reasoning(SimpleNamespace(reqs=[chat, untracked])) is False
    assert sampler.batch_in_reasoning(SimpleNamespace(reqs=[chat, thinking])) is True
    assert untracked.reasoning_phase_state is None
    # Rows leave reasoning independently; the batch follows the last open row.
    thinking.reasoning_phase_state.observe(END)
    assert sampler.batch_in_reasoning(SimpleNamespace(reqs=[chat, thinking])) is False


def test_tracker_without_tokenizer_is_skipped():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=16)
    assert sampler.batch_in_reasoning(SimpleNamespace(reqs=[_req({"end": "</think>"})])) is False


def _server_state(histories, parser="qwen3"):
    return SimpleNamespace(
        config=SimpleNamespace(
            reasoning_parser=parser, reasoning_budget_tokens=0, moe_hot_adapt_histories=histories
        )
    )


def _spec(ctk=None):
    return GenSpec(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs=ctk or {},
    )


def test_phase_spec_only_resolved_for_split3_and_follows_thinking_mode():
    assert resolve_reasoning_phase(_spec(), _server_state("split")) is None
    assert resolve_reasoning_phase(_spec(), _server_state("shared")) is None
    assert resolve_reasoning_phase(_spec(), _server_state("split3", parser=None)) is None
    on = resolve_reasoning_phase(_spec(), _server_state("split3"))
    assert on == {"end": "</think>", "start": "<think>", "open": True}
    off = resolve_reasoning_phase(_spec({"enable_thinking": False}), _server_state("split3"))
    assert off["open"] is False


def _config(**kw):
    return EngineConfig(
        model_path="/tmp/model", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, **kw
    )


def test_config_accepts_split3_and_keeps_default():
    assert _config().moe_hot_adapt_histories == "shared"
    assert _config(moe_hot_adapt_histories="split3").moe_hot_adapt_histories == "split3"
    with pytest.raises(ValueError, match="--moe-hot-adapt-histories"):
        _config(moe_hot_adapt_histories="split4")


def test_cli_flag_accepts_split3():
    from freetoken.server.args import parse_args

    for value in ("split", "split3"):
        args, _ = parse_args([
            "--model", "/tmp/nonexistent-model", "--dtype", "bfloat16",
            "--moe-hot-adapt-histories", value,
        ])
        assert args.moe_hot_adapt_histories == value
    with pytest.raises(SystemExit):
        parse_args([
            "--model", "/tmp/nonexistent-model", "--dtype", "bfloat16",
            "--moe-hot-adapt-histories", "split4",
        ])
