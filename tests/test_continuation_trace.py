"""Hermetic diagnostic checks; GPU tensors are deliberately never accepted."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load("continuation_trace", "python/freetoken/scheduler/continuation_trace.py")
summary = load("continuation_summary", "bench/continuation-trace-summary.py")


class HostIDs:
    is_cpu = True

    def __init__(self, ids):
        self.ids = ids
        self.conversions = 0

    def tolist(self):
        self.conversions += 1
        return list(self.ids)


def request(uid, ids, consumed=None):
    return SimpleNamespace(uid=uid, input_ids=HostIDs(ids), mm_embeds=None,
                           cached_len=len(ids) - 1 if consumed is None else consumed,
                           device_len=len(ids), cache_handle=SimpleNamespace(cached_len=4),
                           mamba_last_track_seqlen=4, toolcall_anchor_len=None)


MANAGER = SimpleNamespace(page_size=4, cache_type="hybrid_radix")


def record_request(writer, uid, prompt, output, cached, *, consumed=None):
    writer.match(request(uid, prompt), cached, MANAGER)
    writer.admitted(uid, len(prompt), cached)
    writer.completed(request(uid, prompt + output, consumed), "stop")


def capture(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    record_request(writer, 1, [1, 2, 3, 4], [5, 6, 7, 8, 9], 0)
    record_request(writer, 2, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11], [12], 4)
    writer.close()
    return writer.path


def rewrite(path, change):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    change(events)
    for i, event in enumerate(events):
        event["seq"] = i
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_disabled_does_not_open_or_construct_a_trace(monkeypatch, tmp_path):
    monkeypatch.delenv(trace.ENV_VAR, raising=False)
    monkeypatch.setattr(trace, "ContinuationTrace", lambda *_: pytest.fail("trace constructed"))
    assert trace.from_env() is None
    monkeypatch.setenv(trace.ENV_VAR, "")
    assert trace.from_env() is None
    assert list(tmp_path.iterdir()) == []


def test_creation_failure_does_not_fail_serving(monkeypatch, tmp_path):
    directory = tmp_path / "file"
    directory.write_text("existing")
    monkeypatch.setenv(trace.ENV_VAR, str(directory))
    assert trace.from_env() is None
    assert directory.read_text() == "existing"


def test_cpu_tokens_are_captured_by_value_and_file_is_private(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    ids = [1, 2, 3, 4]
    req = request(1, ids)
    writer.match(req, 0, MANAGER)
    ids[0] = 99
    writer.close()
    events = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert events[1]["input_ids"] == [1, 2, 3, 4]
    assert req.input_ids.conversions == 1
    assert writer.path.stat().st_mode & 0o777 == 0o600


def test_device_tokens_disable_capture_without_conversion(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    req = request(1, [1, 2])
    req.input_ids.is_cpu = False
    writer.match(req, 0, MANAGER)
    writer.match(req, 0, MANAGER)
    writer.close()
    assert req.input_ids.conversions == 0
    with pytest.raises(ValueError, match="boundaries|footer"):
        summary.read_trace(writer.path)


def test_budget_failure_stops_conversion_and_never_qualifies(tmp_path):
    writer = trace.ContinuationTrace(tmp_path, max_bytes=300)
    req = request(1, list(range(1000)))
    writer.match(req, 0, MANAGER)
    assert writer._failed
    writer.match(req, 0, MANAGER)
    writer.close()
    assert req.input_ids.conversions == 1
    assert writer.path.stat().st_size <= 300
    with pytest.raises(ValueError, match="boundaries|footer"):
        summary.read_trace(writer.path)


def test_write_failure_leaves_incomplete_trace_and_request_untouched(tmp_path, caplog):
    writer = trace.ContinuationTrace(tmp_path)
    writer._file.close()
    req = request(1, [1, 2, 3, 4])
    before = vars(req).copy()
    writer.match(req, 0, MANAGER)
    writer.completed(req, "stop")
    writer.close()
    assert vars(req) == before
    assert req.input_ids.conversions == 1
    assert len(caplog.records) == 1
    with pytest.raises(ValueError, match="boundaries|footer"):
        summary.read_trace(writer.path)


def test_exact_prefix_finds_replayed_generated_tokens(tmp_path):
    result = summary.summarize(capture(tmp_path), [1, 2])
    row = result["transitions"][0]
    assert row["exact_common_prefix_tokens"] == 8
    assert row["first_difference"] == dict(position=8, previous_token=9, next_token=10)
    assert row["matching_generated_tokens_replayed"] == 4
    assert row["aligned_matching_tokens_not_reused"] == 4
    assert result["wall_gate_eligible"] is False


def test_unconsumed_final_sample_is_excluded(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    record_request(writer, 1, [1, 2, 3, 4], [5, 6, 7, 8, 9], 0)
    record_request(writer, 2, list(range(1, 12)), [12], 4)
    writer.close()
    row = summary.summarize(writer.path, [1, 2])["transitions"][0]
    assert row["exact_common_prefix_tokens"] == 9
    assert row["matching_consumed_prefix_upper_bound"] == 8
    assert row["matching_generated_tokens_replayed"] == 4


def test_overlap_device_lead_does_not_invent_matching_host_tokens(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    record_request(writer, 1, [1, 2, 3, 4], [5, 6, 7, 8], 0)
    record_request(writer, 2, list(range(1, 12)), [12], 4)
    writer.close()
    rewrite(writer.path, lambda events: events[3].update(cached_len=10, device_len=11))
    row = summary.summarize(writer.path, [1, 2])["transitions"][0]
    assert row["matching_consumed_prefix_upper_bound"] == 8


def test_rewritten_tool_tokens_limit_reuse(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    record_request(writer, 1, [1, 2, 3, 4], [5, 6, 7, 8, 9], 0)
    record_request(writer, 2, [1, 2, 3, 4, 5, 99, 7, 8, 10], [11], 4)
    writer.close()
    row = summary.summarize(writer.path, [1, 2])["transitions"][0]
    assert row["exact_common_prefix_tokens"] == 5
    assert row["matching_generated_tokens_replayed"] == 1
    assert row["aligned_matching_tokens_not_reused"] == 0


def test_retries_use_last_match_before_successful_admission(tmp_path):
    writer = trace.ContinuationTrace(tmp_path)
    writer.match(request(1, [1, 2, 3, 4]), 0, MANAGER)
    writer.match(request(1, [1, 2, 3, 4]), 2, MANAGER)
    writer.admitted(1, 4, 2)
    writer.completed(request(1, [1, 2, 3, 4, 5]), "stop")
    writer.close()
    requests, metadata = summary.read_trace(writer.path)
    assert requests[1]["match"]["cached_tokens"] == 2
    assert metadata["match_attempts"] == 2
    assert metadata["admitted_requests"] == 1


@pytest.mark.parametrize("change,message", [
    (lambda e: e.pop(), "footer"),
    (lambda e: e.pop(3), "incomplete admitted"),
    (lambda e: e[2].update(cached_tokens=2), "admission differs"),
    (lambda e: e[3].update(input_ids=[99, 2, 3, 4, 5]), "original prompt"),
    (lambda e: e[3].update(cached_len=2), "consumed state"),
    (lambda e: e[3].update(kind="aborted"), "unexpected event"),
    (lambda e: e.insert(4, dict(e[3])), "unique admission"),
])
def test_invalid_capture_cannot_be_used_for_claims(tmp_path, change, message):
    path = capture(tmp_path)
    rewrite(path, change)
    with pytest.raises(ValueError, match=message):
        summary.summarize(path, [1, 2])


@pytest.mark.parametrize("uids,message", [([2, 1], "order"), ([1, 1], "distinct"),
                                          ([1, 3], "missing"), ([1], "two distinct")])
def test_explicit_client_chain_is_required(tmp_path, uids, message):
    with pytest.raises(ValueError, match=message):
        summary.summarize(capture(tmp_path), uids)


def test_multimodal_ids_cannot_prove_continuation(tmp_path):
    path = capture(tmp_path)
    rewrite(path, lambda e: e[1].update(multimodal=True))
    with pytest.raises(ValueError, match="multimodal"):
        summary.summarize(path, [1, 2])


def test_missing_event_and_partial_last_line_are_rejected(tmp_path):
    path = capture(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="truncated"):
        summary.read_trace(path)
    path.write_bytes(raw.replace(b'"seq":2', b'"seq":99'))
    with pytest.raises(ValueError, match="reordered"):
        summary.read_trace(path)
