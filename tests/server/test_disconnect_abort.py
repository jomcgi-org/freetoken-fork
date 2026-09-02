"""GPU-free frontend client-disconnect tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from freetoken.message import AbortMsg
from freetoken.server.api_server import FrontendManager
from freetoken.server.disconnect import DisconnectAwareStreamingResponse


class _AsyncSink:
    def __init__(self) -> None:
        self.items = []

    async def put(self, item) -> None:
        self.items.append(item)

    def stop(self) -> None:
        pass


class _DisconnectRequest:
    async def receive(self):
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}


def _frontend(abort_on_disconnect: str = "on"):
    sink = _AsyncSink()
    state = FrontendManager(
        config=SimpleNamespace(abort_on_disconnect=abort_on_disconnect),
        send_tokenizer=sink,
        recv_tokenizer=SimpleNamespace(stop=lambda: None),
        initialized=True,
        maintenance_state="serving",
    )
    return state, sink


def test_nonstream_disconnect_sends_client_abort():
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()

        async def never_finishes():
            await asyncio.Event().wait()

        with pytest.raises(asyncio.CancelledError):
            await state.run_with_cancellation(
                never_finishes(), _DisconnectRequest(), uid
            )

        assert sink.items == [AbortMsg(uid=uid, client_disconnected=True)]

    asyncio.run(scenario())


def test_stream_disconnect_cancellation_sends_client_abort_before_first_chunk():
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()

        async def no_first_chunk():
            await asyncio.Event().wait()
            yield b"unreachable"

        body = state.stream_with_cancellation(
            no_first_chunk(), _DisconnectRequest(), uid
        )
        response = DisconnectAwareStreamingResponse(body)

        async def send(_message):
            pass

        await response({}, _DisconnectRequest().receive, send)

        assert sink.items == [AbortMsg(uid=uid, client_disconnected=True)]

    asyncio.run(scenario())


def test_disconnect_flag_off_leaves_nonstream_generation_running():
    async def scenario():
        state, sink = _frontend("off")
        uid = state.new_user()

        async def completes():
            return "done"

        assert await state.run_with_cancellation(
            completes(), _DisconnectRequest(), uid
        ) == "done"
        assert sink.items == []

    asyncio.run(scenario())
