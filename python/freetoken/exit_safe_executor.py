"""Thread-pool variant that cannot delay interpreter exit."""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker


class ExitSafeThreadPoolExecutor(ThreadPoolExecutor):
    """Run daemon workers outside concurrent.futures' atexit join registry."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_executor, work_queue=self._work_queue) -> None:
            work_queue.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return
        thread_name = f"{self._thread_name_prefix}_{num_threads}"
        worker = threading.Thread(
            name=thread_name,
            target=_worker,
            args=(
                weakref.ref(self, weakref_cb),
                self._work_queue,
                self._initializer,
                self._initargs,
            ),
            daemon=True,
        )
        worker.start()
        self._threads.add(worker)
        # ThreadPoolExecutor normally adds the worker to the private
        # concurrent.futures.thread._threads_queues registry. Its atexit hook
        # joins every registered thread without a timeout. Omitting that entry
        # means a wedged staging copy or fsync cannot delay process exit. Normal
        # shutdown still joins healthy workers, but process exit may abandon a
        # task and its temporary files instead of completing graceful cleanup.
