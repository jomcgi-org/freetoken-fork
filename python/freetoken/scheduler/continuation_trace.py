"""Opt-in, CPU-only request-boundary tracing for continuation diagnosis."""

import json
import logging
import os
from pathlib import Path
import tempfile


ENV_VAR = "FREETOKEN_CONTINUATION_TRACE_DIR"
FORMAT = "freetoken-continuation-v1"
MAX_BYTES = 64 << 20
logger = logging.getLogger(__name__)


def _cpu_tokens(ids):
    # Never introduce a device transfer or synchronization to obtain diagnostics.
    if not ids.is_cpu:
        raise ValueError("continuation tracing requires existing CPU token IDs")
    return ids.tolist()


class ContinuationTrace:
    def __init__(self, directory, *, max_bytes=MAX_BYTES):
        self._seq = 0
        self._bytes = 0
        self._max_bytes = max_bytes
        self._closed = False
        self._failed = False
        Path(directory).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f"continuation-{os.getpid()}-", suffix=".jsonl",
                                    dir=directory)
        self.path = Path(name)
        self._file = os.fdopen(fd, "wb")
        self._record("header", lambda: dict(format=FORMAT, pid=os.getpid(),
                                           diagnostic=True, wall_gate_eligible=False,
                                           max_bytes=max_bytes))

    def _record(self, kind, fields):
        if self._closed or self._failed:
            return
        try:
            data = dict(seq=self._seq, kind=kind, **fields())
            line = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if self._bytes + len(line) > self._max_bytes:
                raise ValueError("continuation trace byte budget exhausted")
            self._file.write(line)
            self._file.flush()
            self._bytes += len(line)
            self._seq += 1
        except Exception as exc:
            # A missing footer makes partial captures ineligible for analysis.
            # Serving must not fail or change cache ownership when telemetry fails.
            self._failed = True
            logger.warning("Continuation trace disabled; capture is incomplete: %s", exc)

    def match(self, req, cached_tokens, manager):
        self._record("match", lambda: dict(
            uid=req.uid, input_ids=_cpu_tokens(req.input_ids),
            cached_tokens=cached_tokens, multimodal=req.mm_embeds is not None,
            page_size=manager.page_size, cache_type=manager.cache_type))

    def admitted(self, uid, prompt_tokens, cached_tokens):
        self._record("admitted", lambda: dict(uid=uid, prompt_tokens=prompt_tokens,
                                              cached_tokens=cached_tokens))

    def completed(self, req, finish_reason):
        self._record("completed", lambda: dict(
            uid=req.uid, input_ids=_cpu_tokens(req.input_ids), finish_reason=finish_reason,
            cached_len=req.cached_len, device_len=req.device_len,
            cache_handle_len=req.cache_handle.cached_len,
            mamba_last_track_seqlen=req.mamba_last_track_seqlen,
            toolcall_anchor_len=req.toolcall_anchor_len))

    def close(self):
        if self._closed:
            return
        self._record("footer", lambda: dict(complete=True))
        self._closed = True
        try:
            self._file.close()
        except OSError as exc:
            logger.warning("Continuation trace close failed: %s", exc)


def from_env():
    directory = os.environ.get(ENV_VAR)
    if not directory:
        return None
    try:
        return ContinuationTrace(directory)
    except Exception as exc:
        logger.warning("Could not start continuation trace: %s", exc)
        return None
