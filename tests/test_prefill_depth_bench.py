"""Guard the timing and fidelity checks used for the node-4 chunk sweep."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).parents[1] / "bench/prefill-depth.py"
    spec = importlib.util.spec_from_file_location("prefill_depth_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_starts_at_generated_text_and_retains_usage(bench, monkeypatch):
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": '{"r0":'}}]},
        {"choices": [{"delta": {"content": " 1234}"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 8000, "completion_tokens": 7,
                                   "prompt_tokens_details": {"cached_tokens": 64}}},
    ]
    stream = b": keepalive\n\n" + b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    ) + b"data: [DONE]\n\n"
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(stream))
    times = iter([10, 12, 15, 16])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))

    row = bench.request("http://unused", [{"role": "user", "content": "copy"}], 32)

    assert row["ttft_s"] == 2
    assert row["wall_s"] == 6
    assert row["decode_tokens_per_s"] == 2
    assert row["cached_tokens"] == 64
    assert row["chunks"] == events
    assert bench.score(row, {"r0": 1234})


@pytest.mark.parametrize("text", ['{"r0":true}', '{"r0":1,"r0":1}',
                                 '{"r0":1,"extra":2}', '```json\n{"r0":1}\n```'])
def test_fidelity_rejects_wrong_types_duplicate_keys_and_extra_output(bench, text):
    row = dict(text=text, done=True, finish_reason="stop", usage={"completion_tokens": 5},
               error=None, reasoning="")
    assert not bench.score(row, {"r0": 1})


def test_truncated_stream_is_retained_and_fails_fidelity(bench, monkeypatch):
    stream = b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\n'
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(stream))
    row = bench.request("http://unused", [], 32)
    assert row["text"] == "{}"
    assert not row["done"]
    assert not bench.score(row, {})
