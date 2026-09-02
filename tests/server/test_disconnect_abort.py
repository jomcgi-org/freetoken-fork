"""GPU-free frontend client-disconnect tests."""

from __future__ import annotations

import asyncio
import gc
import multiprocessing
import multiprocessing.util
import threading
from types import SimpleNamespace

import pytest
import torch  # noqa: F401 -- importing the GPU-free torch build is a module-load prerequisite

from freetoken.message import AbortMsg, UserReply
from freetoken.server import api_server
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.api_server import FrontendManager
from freetoken.server.disconnect import (
    ClientDisconnectedResponse,
    DisconnectAwareStreamingResponse,
)
from freetoken.server.openai_api import handle_chat_completion
from freetoken.server.supervisor import BackendHandle, LoadProgress, run_backend_supervisor


class _AsyncSink:
    def __init__(self) -> None:
        self.items = []

    async def put(self, item) -> None:
        self.items.append(item)

    def stop(self) -> None:
        pass


class _DisconnectRequest:
    def __init__(self, on_disconnect=None, disconnect_after=None) -> None:
        self.headers = {}
        self.disconnected = False
        self.on_disconnect = on_disconnect
        self.disconnect_after = disconnect_after

    async def receive(self):
        if self.disconnect_after is not None:
            await self.disconnect_after.wait()
        await asyncio.sleep(0)
        self.disconnected = True
        if self.on_disconnect is not None:
            self.on_disconnect()
        return {"type": "http.disconnect"}


class _ConnectedRequest:
    headers = {}

    async def receive(self):
        await asyncio.Event().wait()


def _frontend(abort_on_disconnect: str = "on"):
    sink = _AsyncSink()
    state = FrontendManager(
        config=SimpleNamespace(
            abort_on_disconnect=abort_on_disconnect,
            model_path="/models/unit-model",
            reasoning_parser=None,
            tool_call_parser="llama3",
        ),
        send_tokenizer=sink,
        recv_tokenizer=SimpleNamespace(stop=lambda: None),
        initialized=True,
        maintenance_state="serving",
    )
    return state, sink


def _capture_info(monkeypatch):
    lines = []
    monkeypatch.setattr(
        api_server.logger,
        "info",
        lambda message, *args: lines.append(message % args),
    )
    return lines


def _deliver_reply(state, reply: UserReply) -> None:
    """Mirror FrontendManager.listen for one backend reply without starting its task."""

    state.stats.observe(reply)
    state._generated_tokens[reply.uid] = (
        state._generated_tokens.get(reply.uid, 0) + reply.completion_tokens_delta
    )
    if reply.finished:
        state._abort_sent.discard(reply.uid)
        state._generated_tokens.pop(reply.uid, None)
    state.ack_map[reply.uid].append(reply)
    state.event_map[reply.uid].set()


def test_nonstream_disconnect_returns_empty_499_and_aborts_once(monkeypatch):
    async def scenario():
        state, sink = _frontend()
        request = _DisconnectRequest(
            on_disconnect=lambda: state._generated_tokens.__setitem__(0, 3)
        )
        response = await handle_chat_completion(
            ChatCompletionRequest(
                model="unit-model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=8,
            ),
            request=request,
            state=state,
            model_sampling={},
        )

        assert isinstance(response, ClientDisconnectedResponse)
        assert response.status_code == 499
        assert response.body == b""
        aborts = [item for item in sink.items if isinstance(item, AbortMsg)]
        assert aborts == [AbortMsg(uid=0, client_disconnected=True)]

        await state.abort_user(0, client_disconnected=True)
        assert [item for item in sink.items if isinstance(item, AbortMsg)] == aborts

    info_lines = _capture_info(monkeypatch)
    asyncio.run(scenario())
    assert info_lines == [
        "Client disconnected: request_id=0 tokens_generated=3 scheduler_abort=issued"
    ]


def test_stream_terminal_ack_then_disconnect_sends_no_late_abort():
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()
        _deliver_reply(
            state,
            UserReply(
                uid=uid,
                incremental_output="done",
                finished=True,
                completion_tokens_delta=1,
            ),
        )

        token_sent = asyncio.Event()
        request = _DisconnectRequest(disconnect_after=token_sent)
        body = state.stream_with_cancellation(
            state.stream_generate(uid), request, uid
        )
        response = DisconnectAwareStreamingResponse(body, request=request)
        sent = []

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                token_sent.set()
                await asyncio.sleep(0)

        await response({}, request.receive, send)

        assert any(message["type"] == "http.response.body" for message in sent)
        assert not [item for item in sink.items if isinstance(item, AbortMsg)]
        assert uid not in state._generated_tokens

        await state.abort_user(uid, client_disconnected=True)
        assert not [item for item in sink.items if isinstance(item, AbortMsg)]

    asyncio.run(scenario())


def test_stream_midstream_disconnect_aborts_exactly_once(monkeypatch):
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()
        _deliver_reply(
            state,
            UserReply(
                uid=uid,
                incremental_output="partial",
                finished=False,
                completion_tokens_delta=2,
            ),
        )

        token_sent = asyncio.Event()
        request = _DisconnectRequest(disconnect_after=token_sent)
        body = state.stream_with_cancellation(
            state.stream_generate(uid), request, uid
        )
        response = DisconnectAwareStreamingResponse(body, request=request)

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                token_sent.set()

        await response({}, request.receive, send)

        aborts = [item for item in sink.items if isinstance(item, AbortMsg)]
        assert aborts == [AbortMsg(uid=uid, client_disconnected=True)]
        assert uid not in state._generated_tokens

        await state.abort_user(uid, client_disconnected=True)
        assert [item for item in sink.items if isinstance(item, AbortMsg)] == aborts

    info_lines = _capture_info(monkeypatch)
    asyncio.run(scenario())
    assert info_lines == [
        "Client disconnected: request_id=0 tokens_generated=2 scheduler_abort=issued"
    ]


def test_server_side_stream_cancellation_is_propagated_without_abort():
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()
        body = state.stream_with_cancellation(
            state.stream_generate(uid), _ConnectedRequest(), uid
        )
        pending = asyncio.create_task(anext(body))
        await asyncio.sleep(0)
        pending.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pending

        assert not [item for item in sink.items if isinstance(item, AbortMsg)]
        assert uid not in state._generated_tokens

    asyncio.run(scenario())


def test_normal_nonstream_request_is_unchanged(monkeypatch):
    async def scenario():
        state, sink = _frontend()
        uid = state.new_user()
        result = {"content": "done"}

        async def completes():
            return result

        assert await state.run_with_cancellation(
            completes(), _ConnectedRequest(), uid
        ) is result
        assert sink.items == []

    info_lines = _capture_info(monkeypatch)
    asyncio.run(scenario())
    assert info_lines == []


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


def _semlock_finalizer_count() -> int:
    return sum(
        "SemLock" in repr(getattr(finalizer, "_callback", None))
        for finalizer in multiprocessing.util._finalizer_registry.values()
    )


def test_shutdown_stops_supervisor_and_releases_ack_queue_semlocks():
    class _AliveProcess:
        name = "test-backend"

        def is_alive(self) -> bool:
            return True

    state, _sink = _frontend()
    state.backend_handle = BackendHandle(
        ack_queue=multiprocessing.get_context("spawn").Queue(),
        processes=[_AliveProcess()],
        expected_acks=0,
    )
    state.backend_supervisor_stop = threading.Event()
    ready = threading.Event()
    supervisor = threading.Thread(
        target=run_backend_supervisor,
        args=(state.backend_handle, LoadProgress(), ready.set),
        kwargs={"stop_event": state.backend_supervisor_stop, "poll": 0.01},
        daemon=True,
    )
    state.backend_supervisor_thread = supervisor
    supervisor.start()
    assert ready.wait(timeout=1.0)
    assert _semlock_finalizer_count() == 3

    state.shutdown()
    gc.collect()

    assert not supervisor.is_alive()
    assert state.backend_supervisor_thread is None
    assert state.backend_handle is None
    assert _semlock_finalizer_count() == 0
