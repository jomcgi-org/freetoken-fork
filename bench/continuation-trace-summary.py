"""Compare explicitly selected turns in a complete diagnostic token capture."""

import argparse
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _tokens(value):
    return isinstance(value, list) and all(_integer(token) for token in value)


def read_trace(path):
    raw = Path(path).read_bytes()
    _require(raw.endswith(b"\n"), "truncated trace")
    events = [json.loads(line) for line in raw.splitlines()]
    _require(len(events) >= 2, "missing trace boundaries")
    _require(all(event.get("seq") == i for i, event in enumerate(events)),
             "missing or reordered events")
    header, footer = events[0], events[-1]
    _require(header.get("kind") == "header"
             and header.get("format") == "freetoken-continuation-v1"
             and header.get("diagnostic") is True
             and header.get("wall_gate_eligible") is False, "unsupported trace header")
    _require(footer.get("kind") == "footer" and footer.get("complete") is True,
             "incomplete capture: missing footer")
    attempts, requests = {}, {}
    match_count = 0
    for event in events[1:-1]:
        uid, kind = event.get("uid"), event.get("kind")
        _require(_integer(uid), "invalid request uid")
        if kind == "match":
            _require(uid not in requests, "match after admission")
            ids = event.get("input_ids")
            _require(_tokens(ids) and len(ids) > 0, "invalid prompt tokens")
            _require(_integer(event.get("cached_tokens"))
                     and event["cached_tokens"] < len(ids), "invalid matched length")
            _require(_integer(event.get("page_size"), 1), "invalid page size")
            _require(type(event.get("multimodal")) is bool, "missing multimodal status")
            attempts[uid] = event
            match_count += 1
        elif kind == "admitted":
            _require(uid in attempts and uid not in requests, "admission without unique match")
            match = attempts.pop(uid)
            _require(event.get("prompt_tokens") == len(match["input_ids"])
                     and event.get("cached_tokens") == match["cached_tokens"],
                     "admission differs from last match")
            requests[uid] = dict(match=match, admitted=event)
        elif kind == "completed":
            _require(uid in requests and "completed" not in requests[uid],
                     "completion without unique admission")
            request = requests[uid]
            prompt = request["match"]["input_ids"]
            ids = event.get("input_ids")
            _require(_tokens(ids) and len(ids) > len(prompt) and ids[:len(prompt)] == prompt,
                     "completion does not extend original prompt")
            _require(event.get("finish_reason") in ("stop", "length"), "invalid completion")
            _require(_integer(event.get("cached_len"), len(prompt))
                     and _integer(event.get("device_len"), event["cached_len"]),
                     "invalid consumed state boundary")
            request["completed"] = event
        else:
            raise ValueError(f"unexpected event kind: {kind}")
    _require(all("completed" in request for request in requests.values()),
             "incomplete admitted requests (including aborts)")
    return requests, dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                          pid=header["pid"], admitted_requests=len(requests),
                          match_attempts=match_count, unmatched_request_uids=sorted(attempts))


def summarize(path, uids):
    requests, capture = read_trace(path)
    _require(len(uids) >= 2 and len(set(uids)) == len(uids),
             "select at least two distinct uids from one continuing client session")
    _require(all(uid in requests for uid in uids), "selected request missing")
    transitions = []
    for previous_uid, next_uid in zip(uids, uids[1:]):
        previous, following = requests[previous_uid], requests[next_uid]
        before, after = previous["completed"], following["match"]
        _require(before["seq"] < after["seq"], "selected turns overlap or are out of order")
        _require(not previous["match"]["multimodal"] and not after["multimodal"],
                 "token identity cannot qualify multimodal continuation")
        _require(previous["match"]["page_size"] == after["page_size"]
                 and previous["match"]["cache_type"] == after["cache_type"],
                 "cache geometry changed")
        old_ids, new_ids = before["input_ids"], after["input_ids"]
        lcp = 0
        for old, new in zip(old_ids, new_ids):
            if old != new:
                break
            lcp += 1
        prompt_len = len(previous["match"]["input_ids"])
        cached = following["admitted"]["cached_tokens"]
        # The final sampled token may not have been consumed. Overlap may also
        # advance device metadata ahead of host output. Neither is a snapshot.
        consumed_match = min(lcp, before["cached_len"], len(new_ids) - 1)
        aligned_match = consumed_match // after["page_size"] * after["page_size"]
        transitions.append(dict(
            previous_uid=previous_uid, next_uid=next_uid,
            previous_prompt_tokens=prompt_len, previous_host_tokens=len(old_ids),
            previous_cached_len=before["cached_len"], previous_device_len=before["device_len"],
            previous_cache_handle_len=before["cache_handle_len"],
            previous_mamba_last_track_seqlen=before["mamba_last_track_seqlen"],
            previous_toolcall_anchor_len=before["toolcall_anchor_len"],
            next_prompt_tokens=len(new_ids), next_cached_tokens=cached,
            exact_common_prefix_tokens=lcp,
            divergence_before_previous_prompt_end=lcp < prompt_len,
            first_difference=(dict(position=lcp, previous_token=old_ids[lcp],
                                   next_token=new_ids[lcp])
                              if lcp < min(len(old_ids), len(new_ids)) else None),
            matching_consumed_prefix_upper_bound=consumed_match,
            matching_consumed_tokens_not_reused=max(0, consumed_match - cached),
            matching_generated_tokens_replayed=max(0, consumed_match - max(cached, prompt_len)),
            aligned_matching_tokens_not_reused=max(0, aligned_match - cached)))
    return dict(diagnostic=True, wall_gate_eligible=False, capture=capture, selected_uids=uids,
                transitions=transitions,
                limitations=["Selected uids must come from the same continuing client session.",
                             "Token identity does not prove KV or recurrent state remains available.",
                             "Consumed lengths are host metadata, not synchronized state snapshots.",
                             "No additional cache walks, saved wall time, or quality claim."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--uids", required=True,
                        help="Comma-separated numeric uids, in client conversation order")
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace, [int(uid) for uid in args.uids.split(",")]), indent=2))


if __name__ == "__main__":
    main()
