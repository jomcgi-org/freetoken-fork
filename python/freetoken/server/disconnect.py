"""ASGI response primitives with eager peer-disconnect detection."""

from __future__ import annotations

import anyio
from fastapi.responses import Response, StreamingResponse


_DISCONNECTED_ATTR = "_freetoken_client_disconnected"


def client_disconnected(request) -> bool:
    """Whether this response's ASGI watcher observed ``http.disconnect``."""

    return bool(getattr(request, _DISCONNECTED_ATTR, False))


class ClientDisconnectedResponse(Response):
    """Empty terminal response for a peer that has already closed its request."""

    def __init__(self) -> None:
        super().__init__(content=b"", status_code=499)


class DisconnectAwareStreamingResponse(StreamingResponse):
    """Always listen for http.disconnect while the body iterator is blocked.

    Some Starlette versions use send failures instead of a receive listener for newer ASGI
    specs. A generation stream waiting on queueing or prefill sends nothing, so that mode
    cannot notice the lost peer. This keeps one receive listener active for every ASGI version.
    """

    def __init__(self, *args, request=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.request = request
        if request is not None:
            setattr(request, _DISCONNECTED_ATTR, False)

    async def __call__(self, scope, receive, send) -> None:
        disconnected = anyio.Event()

        # Start the ASGI response before racing the body against peer disconnect. This gives
        # middleware a complete response cycle even if the disconnect is already buffered.
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )

        async def listen_for_disconnect() -> None:
            await self.listen_for_disconnect(receive)
            if self.request is not None:
                # Publish the cause before cancelling stream_body. Its cancellation handler
                # must not mistake an unrelated server-side cancellation for a lost client.
                setattr(self.request, _DISCONNECTED_ATTR, True)
            disconnected.set()

        async def stream_body() -> None:
            async for chunk in self.body_iterator:
                if disconnected.is_set():
                    return
                if not isinstance(chunk, (bytes, memoryview)):
                    chunk = chunk.encode(self.charset)
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
            if not disconnected.is_set():
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )

        async with anyio.create_task_group() as task_group:
            async def run_until_complete(func) -> None:
                await func()
                task_group.cancel_scope.cancel()

            task_group.start_soon(run_until_complete, stream_body)
            await run_until_complete(listen_for_disconnect)

        if self.background is not None:
            await self.background()
