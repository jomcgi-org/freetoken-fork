from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.priority import resolve_request_priority
from freetoken.server.responses_api import ResponsesRequest, convert_responses_to_genspec


def test_openai_request_bodies_accept_integer_priority():
    chat = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], priority=4
    )
    completion = CompletionRequest(model="m", prompt="hi", priority=-2)
    responses = ResponsesRequest(model="m", input="hi", priority=7)

    assert (chat.priority, completion.priority, responses.priority) == (4, -2, 7)
    assert convert_responses_to_genspec(responses, {}).priority == 7


def test_priority_header_wins_over_body_and_both_default_to_zero():
    assert resolve_request_priority() == 0
    assert resolve_request_priority(4, SimpleNamespace(headers={})) == 4
    request = SimpleNamespace(headers={"x-request-priority": "9"})
    assert resolve_request_priority(4, request) == 9


def test_invalid_priority_header_is_rejected():
    request = SimpleNamespace(headers={"x-request-priority": "urgent"})
    with pytest.raises(ValueError, match="must be an integer"):
        resolve_request_priority(4, request)
