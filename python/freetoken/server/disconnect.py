"""ASGI response primitives with eager peer-disconnect detection."""

from __future__ import annotations

from functools import partial

import anyio
from fastapi.responses import StreamingResponse


class DisconnectAwareStreamingResponse(StreamingResponse):
    """Always listen for http.disconnect while the body iterator is blocked.

    Some Starlette versions use send failures instead of a receive listener for newer ASGI
    specs. A generation stream waiting on queueing or prefill sends nothing, so that mode
    cannot notice the lost peer. This keeps one receive listener active for every ASGI version.
    """

    async def __call__(self, scope, receive, send) -> None:
        async with anyio.create_task_group() as task_group:
            async def run_until_complete(func) -> None:
                await func()
                task_group.cancel_scope.cancel()

            task_group.start_soon(
                run_until_complete, partial(self.stream_response, send)
            )
            await run_until_complete(partial(self.listen_for_disconnect, receive))

        if self.background is not None:
            await self.background()
