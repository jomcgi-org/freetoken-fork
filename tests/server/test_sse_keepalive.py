"""Keep-alive frames on the streaming routes: a long silent prefill must not leave the
wire idle (Node undici closes a response body that is silent for 300 s)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from freetoken.message import UserReply
from freetoken.server import generation
from freetoken.server.generation import (
    DEFAULT_KEEPALIVE_INTERVAL_S,
    KEEPALIVE_ENV,
    keepalive_interval_s,
)
from freetoken.server.openai_api import ChatCompletionRequest, stream_chat_completion_chunks

from .test_openai_api import parse_sse

KEEPALIVE_FRAME = b": keepalive\n\n"


class SlowState:
    """Engine stand-in whose first reply arrives only after ``delay`` seconds (prefill)."""

    def __init__(self, delay: float) -> None:
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=None,
        )
        self.delay = delay

    async def send_one(self, msg):
        pass

    async def wait_for_ack(self, uid: int):
        await asyncio.sleep(self.delay)
        yield UserReply(uid=uid, incremental_output="Hello", finished=False)
        yield UserReply(uid=uid, incremental_output=" world", finished=True)


def _collect_chat(delay: float) -> list[bytes]:
    req = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=8
    )

    async def go():
        return [c async for c in stream_chat_completion_chunks(42, req, SlowState(delay))]

    return asyncio.run(go())


def _content(chunks: list[bytes]) -> str:
    return "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in parse_sse(chunks)
        if isinstance(c, dict) and c.get("choices")
    )


def test_chat_stream_emits_keepalive_comment_before_first_token(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_ENV, "0.05")
    chunks = _collect_chat(delay=0.2)

    assert KEEPALIVE_FRAME in chunks
    first_keepalive = chunks.index(KEEPALIVE_FRAME)
    first_token = next(i for i, c in enumerate(chunks) if b'"content": "Hello"' in c)
    assert first_keepalive < first_token
    # Only the role chunk may precede the first keep-alive.
    assert first_keepalive == 1
    assert _content(chunks) == "Hello world"
    assert parse_sse(chunks)[-1] == "[DONE]"


def test_chat_stream_output_unchanged_apart_from_keepalives(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_ENV, "0.05")
    with_ka = _collect_chat(delay=0.2)
    monkeypatch.setenv(KEEPALIVE_ENV, "0")
    without_ka = _collect_chat(delay=0.2)

    assert KEEPALIVE_FRAME not in without_ka
    stripped = [c for c in with_ka if c != KEEPALIVE_FRAME]
    # Frames embed id/created; compare the parsed payloads minus the timestamp.
    def norm(frames):
        return [
            {k: v for k, v in e.items() if k != "created"} if isinstance(e, dict) else e
            for e in parse_sse(frames)
        ]

    assert norm(stripped) == norm(without_ka)


def test_chat_stream_keepalive_disabled_with_zero(monkeypatch):
    monkeypatch.setenv(KEEPALIVE_ENV, "0")
    chunks = _collect_chat(delay=0.15)
    assert KEEPALIVE_FRAME not in chunks
    assert _content(chunks) == "Hello world"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_KEEPALIVE_INTERVAL_S),
        ("", DEFAULT_KEEPALIVE_INTERVAL_S),
        ("junk", DEFAULT_KEEPALIVE_INTERVAL_S),
        ("-3", DEFAULT_KEEPALIVE_INTERVAL_S),
        ("0", 0.0),
        ("2.5", 2.5),
    ],
)
def test_keepalive_interval_env_override(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(KEEPALIVE_ENV, raising=False)
    else:
        monkeypatch.setenv(KEEPALIVE_ENV, raw)
    assert keepalive_interval_s() == expected
    assert DEFAULT_KEEPALIVE_INTERVAL_S == 15.0


def test_with_keepalive_zero_interval_is_passthrough():
    async def slow():
        await asyncio.sleep(0.05)
        yield 1

    async def go():
        return [ev async for ev in generation.with_keepalive(slow(), 0)]

    assert asyncio.run(go()) == [1]


def test_generate_route_streams_keepalive_during_silence(monkeypatch):
    from fastapi.testclient import TestClient
    from freetoken.server import api_server

    monkeypatch.setenv(KEEPALIVE_ENV, "0.05")

    class State:
        maintenance_state = "serving"

        def new_user(self):
            return 7

        async def send_one(self, msg):
            pass

        async def stream_generate(self, uid):
            await asyncio.sleep(0.2)
            yield b'data: {"text": "hi"}\n\n'
            yield b"data: [DONE]\n\n"

        async def stream_with_cancellation(self, generator, request, uid):
            async for chunk in generator:
                yield chunk

    monkeypatch.setattr(api_server, "get_global_state", lambda: State())
    # No ``with``: skip the app lifespan (it would shut down the real global state).
    client = TestClient(api_server.app)
    body = client.post("/generate", json={"prompt": "x", "max_tokens": 4}).content

    assert body.startswith(KEEPALIVE_FRAME)
    assert body.endswith(b'data: {"text": "hi"}\n\ndata: [DONE]\n\n')
