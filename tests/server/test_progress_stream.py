"""Pre-commit HTTP errors and opt-in progress frames on the three streaming routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from freetoken.message import UserReply
from freetoken.server.anthropic_api import handle_anthropic_messages
from freetoken.server.generation import ADMISSION_WAIT_ENV, prime_events, GenerationError
from freetoken.server.openai_api import ChatCompletionRequest, handle_chat_completion
from freetoken.server.responses_api import ResponsesRequest, handle_responses

PROGRESS = {"x-freetoken-include-progress": "1"}
TOO_LONG = "prompt is too long: 9000 tokens > 4096 maximum (prompt + generation)"


class State:
    def __init__(self, replies, delay: float = 0.0) -> None:
        self.config = SimpleNamespace(
            model_path="/models/unit-model", served_model_name="unit-model",
            tool_call_parser="llama3", reasoning_parser=None,
        )
        self.replies = replies
        self.delay = delay

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        pass

    def stream_with_cancellation(self, generator, request, uid):
        return generator

    async def wait_for_ack(self, uid: int):
        for r in self.replies:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield r


def _request(headers=None):
    return SimpleNamespace(headers=headers or {})


def _ok_replies():
    return [
        UserReply(uid=42, incremental_output="", finished=False,
                  prompt_tokens_delta=100, cached_tokens=64),
        UserReply(uid=42, incremental_output="Hi", finished=False, completion_tokens_delta=1),
        UserReply(uid=42, incremental_output="!", finished=True, completion_tokens_delta=1,
                  finish_reason="stop"),
    ]


def _err_replies(status=400, code="context_length_exceeded", msg=TOO_LONG):
    return [UserReply(uid=42, incremental_output="", finished=True, error=msg,
                      error_code=code, error_status_code=status)]


def _call(route, stream, state, headers=None):
    req = _request(headers)
    if route == "chat":
        body = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=8, stream=stream)
        coro = handle_chat_completion(body, req, state, {})
    elif route == "responses":
        coro = handle_responses(ResponsesRequest(model="m", input="hi", stream=stream),
                                req, state, {})
    else:
        from freetoken.server.anthropic_models import AnthropicMessagesRequest
        body = AnthropicMessagesRequest(
            model="m", max_tokens=8, stream=stream,
            messages=[{"role": "user", "content": "hi"}])
        coro = handle_anthropic_messages(body, req, state, {})
    return coro


async def _run(route, stream, state, headers=None):
    resp = await _call(route, stream, state, headers)
    if not stream or resp.status_code != 200 or not hasattr(resp, "body_iterator"):
        return resp, None
    chunks = []
    async for c in resp.body_iterator:
        chunks.append(c if isinstance(c, str) else c.decode())
    return resp, "".join(chunks)


ROUTES = ["chat", "responses", "anthropic"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_too_long_prompt_is_http_400(route, stream):
    resp, body = asyncio.run(_run(route, stream, State(_err_replies())))
    assert resp.status_code == 400
    assert body is None  # never committed an SSE stream
    assert "prompt is too long" in resp.body.decode()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_capacity_refusal_is_503_with_retry_after(route, stream):
    state = State(_err_replies(503, "server_overloaded", "out of memory; retry"))
    resp, body = asyncio.run(_run(route, stream, state))
    assert resp.status_code == 503
    assert resp.headers["retry-after"]
    assert body is None


@pytest.mark.parametrize("route", ROUTES)
def test_failure_after_first_event_stays_in_band(route):
    replies = [
        UserReply(uid=42, incremental_output="Hi", finished=False, completion_tokens_delta=1),
        *_err_replies(503, "server_overloaded", "out of memory; retry"),
    ]
    resp, body = asyncio.run(_run(route, True, State(replies)))
    assert resp.status_code == 200
    assert "out of memory" in body


@pytest.mark.parametrize("route", ROUTES)
def test_slow_engine_still_commits_after_wait_window(route, monkeypatch):
    monkeypatch.setenv(ADMISSION_WAIT_ENV, "0.05")
    resp, body = asyncio.run(_run(route, True, State(_ok_replies(), delay=0.2)))
    assert resp.status_code == 200
    assert "Hi" in body


@pytest.mark.parametrize("route", ROUTES)
def test_progress_frames_only_with_header(route):
    _, plain = asyncio.run(_run(route, True, State(_ok_replies())))
    _, prog = asyncio.run(_run(route, True, State(_ok_replies()), PROGRESS))
    assert ": progress" not in plain
    lines = [l for l in prog.splitlines() if l.startswith(": progress")]
    assert lines == [
        ': progress {"stage":"queued"}',
        ': progress {"stage":"prefill","done":64,"total":100,"reused":64}',
        ': progress {"stage":"generating"}',
    ]
    # Output is otherwise unchanged (ids/timestamps aside): same frame count minus comments.
    stripped = "\n".join(l for l in prog.splitlines() if not l.startswith(": progress"))
    assert stripped.count("data:") == plain.count("data:")


def test_prime_events_replays_first_event_and_rest():
    async def gen():
        yield 1
        yield 2

    async def go():
        err, it = await prime_events(gen(), 1.0)
        assert err is None
        return [x async for x in it]

    assert asyncio.run(go()) == [1, 2]


def test_prime_events_zero_wait_is_passthrough():
    async def gen():
        raise GenerationError("boom")
        yield

    async def go():
        err, it = await prime_events(gen(), 0)
        assert err is None
        with pytest.raises(GenerationError):
            async for _ in it:
                pass

    asyncio.run(go())
