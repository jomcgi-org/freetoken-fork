"""Reasoning-budget wire parsing for all three protocols and server-default resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.core import SamplingParams
from freetoken.server import anthropic_api as A
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.generation import GenSpec, resolve_reasoning_budget
from freetoken.server.openai_api import chat_request_to_genspec
from freetoken.server.responses_api import ResponsesRequest, convert_responses_to_genspec


def _chat(**extra):
    return ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
    )


def _chat_budget(**extra):
    return chat_request_to_genspec(_chat(**extra), {}, SimpleNamespace()).reasoning_budget_tokens


def test_chat_absent_is_none():
    assert _chat_budget() is None
    assert _chat_budget(reasoning_effort="high") is None  # never derived from effort


def test_chat_max_reasoning_tokens():
    assert _chat_budget(max_reasoning_tokens=32) == 32


def test_chat_reasoning_max_tokens():
    assert _chat_budget(reasoning={"max_tokens": 64, "effort": "low"}) == 64


def test_chat_max_reasoning_tokens_wins_over_reasoning_object():
    assert _chat_budget(max_reasoning_tokens=8, reasoning={"max_tokens": 64}) == 8


def test_chat_zero_is_explicit_unlimited():
    assert _chat_budget(max_reasoning_tokens=0) == 0


@pytest.mark.parametrize("bad", [-1, True, "7", 1.5])
def test_chat_rejects_invalid(bad):
    with pytest.raises(ValueError):
        chat_request_to_genspec(
            _chat(reasoning={"max_tokens": bad}), {}, SimpleNamespace()
        )


def test_responses_reasoning_max_tokens():
    req = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "high", "max_tokens": 48}}
    )
    assert convert_responses_to_genspec(req, {}).reasoning_budget_tokens == 48


def test_responses_effort_alone_gives_no_budget():
    req = ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": "high"}}
    )
    assert convert_responses_to_genspec(req, {}).reasoning_budget_tokens is None


def _anthropic(thinking):
    body = {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    if thinking is not None:
        body["thinking"] = thinking
    return A.convert_anthropic_to_genspec(AnthropicMessagesRequest.model_validate(body), {})


def test_anthropic_budget_tokens():
    spec = _anthropic({"type": "enabled", "budget_tokens": 2048})
    assert spec.reasoning_budget_tokens == 2048


def test_anthropic_disabled_or_absent_has_no_budget():
    assert _anthropic({"type": "disabled", "budget_tokens": 2048}).reasoning_budget_tokens is None
    assert _anthropic(None).reasoning_budget_tokens is None


def _state(parser="qwen3", default=0):
    return SimpleNamespace(
        config=SimpleNamespace(reasoning_parser=parser, reasoning_budget_tokens=default)
    )


def _spec(tokens=None, ctk=None):
    return GenSpec(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs=ctk or {},
        reasoning_budget_tokens=tokens,
    )


def test_default_zero_is_unlimited():
    assert resolve_reasoning_budget(_spec(), _state()) is None


def test_server_default_applies_and_uses_parser_tags():
    budget = resolve_reasoning_budget(_spec(), _state(default=100))
    assert budget == {"tokens": 100, "end": "</think>", "start": "<think>", "open": True}


def test_request_overrides_default_including_zero():
    assert resolve_reasoning_budget(_spec(10), _state(default=100))["tokens"] == 10
    assert resolve_reasoning_budget(_spec(0), _state(default=100)) is None


def test_open_follows_template_thinking_mode():
    budget = resolve_reasoning_budget(
        _spec(5, {"enable_thinking": False}), _state(default=0)
    )
    assert budget["open"] is False  # model must open <think> itself before counting


def test_minimax_m3_tags():
    budget = resolve_reasoning_budget(
        _spec(5, {"thinking_mode": "enabled"}), _state("minimax_m3")
    )
    assert (budget["start"], budget["end"], budget["open"]) == ("<mm:think>", "</mm:think>", True)


def test_no_reasoning_parser_or_tagless_parser_is_unbudgeted():
    assert resolve_reasoning_budget(_spec(5), _state(parser=None)) is None
    assert resolve_reasoning_budget(_spec(5), _state(parser="gpt_oss")) is None
