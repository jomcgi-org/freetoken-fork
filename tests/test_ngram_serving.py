"""Causal drafts and target ownership preserve exact committed-prefix semantics."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch

from freetoken.verification.ngram import propose, proposal_for_request
from freetoken.verification.runtime import NgramTarget
from freetoken.verification import runtime


def reference(tokens, match, drafts, lookback):
    values = tokens[-lookback:]
    suffix = values[-match:]
    for start in range(len(values) - match - drafts, -1, -1):
        if values[start:start + match] == suffix:
            return values[start + match:start + match + drafts]
    return None


def test_byte_search_matches_token_reference_for_every_causal_prefix():
    rng = random.Random(173)
    for alphabet in (3, 256, 2**31):
        tokens = [rng.randrange(alphabet) for _ in range(45)]
        tokens += tokens[5:20] + tokens[2:9]
        for match in (1, 3, 8):
            for end in range(1, len(tokens) + 1):
                prefix = tokens[:end]
                assert propose(prefix, match=match, drafts=4, lookback=32) == reference(prefix, match, 4, 32)


def test_proposer_uses_recent_complete_known_continuation_only():
    suffix = list(range(8))
    assert propose(suffix + [31, 32, 33, 34] + suffix) == [31, 32, 33, 34]
    assert propose(suffix + [31, 32, 33]) is None
    assert propose(suffix + [31, 32, 33, 34] + [999] * 24 + suffix, lookback=24) is None
    assert propose([1] * 12) == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        propose([], lookback=1)


def req_fixture():
    history = list(range(8)) + [31, 32, 33, 34] + list(range(8))
    return SimpleNamespace(input_ids=torch.tensor(history), cached_len=len(history) - 1,
                           device_len=len(history), remain_len=20, toolcall_anchor_len=None,
                           sampling_params=SimpleNamespace(is_greedy=True, guided_decoding=None))


@pytest.mark.parametrize("reason", ["budget", "host_lag", "extend", "sampling", "grammar",
                                   "dormant_grammar", "multimodal", "lazy", "anchor"])
def test_unsupported_request_falls_back_without_mutation(reason):
    req = req_fixture()
    assert proposal_for_request(req) == [31, 32, 33, 34]
    if reason == "budget": req.remain_len = 4
    elif reason == "host_lag": req.input_ids = req.input_ids[:-1]
    elif reason == "extend": req.cached_len -= 1
    elif reason == "sampling": req.sampling_params.is_greedy = False
    elif reason == "grammar": req.sampling_params.guided_decoding = {}
    elif reason == "dormant_grammar": req.guided_state = object()
    elif reason == "multimodal": req.mm_embeds = object()
    elif reason == "lazy": req.lazy_kv_restore = SimpleNamespace(complete=False)
    else: req.toolcall_anchor_len = req.cached_len + 1
    before = (req.cached_len, req.device_len, req.input_ids.clone())
    assert proposal_for_request(req) is None
    assert (req.cached_len, req.device_len) == before[:2]
    assert torch.equal(req.input_ids, before[2])


@pytest.fixture
def target(monkeypatch):
    # Runtime ownership tests isolate execution; numerical CUDA graph parity is
    # covered by the shared target tests and the exclusive full-model gate.
    monkeypatch.setitem(sys.modules, "freetoken.attention.linear",
                        SimpleNamespace(build_fla_metadata=lambda batch, device: None))
    engine = SimpleNamespace(device="cpu", config=SimpleNamespace(ngram_debug=False),
                             attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None))
    target = NgramTarget(engine)
    restored = []
    target.checkpoint = SimpleNamespace(restore=restored.append)
    req = SimpleNamespace(cached_len=10, device_len=15)
    batch = SimpleNamespace(reqs=[req], padded_reqs=[req], input_ids=torch.tensor([9, 0, 0, 0, 0]),
                            positions=torch.arange(10, 15), out_loc=torch.arange(90, 95),
                            active_table_idx=torch.tensor([2]), phase="decode",
                            mtp_original_cached_len=10, mtp_original_device_len=11,
                            ngram_drafts=[11, 12, 13, 14], ngram_interrupt_ids=())
    def set_predictions(tokens):
        logits = torch.full((5, 32), -100.)
        logits[torch.arange(5), tokens] = 100.
        target.graph = SimpleNamespace(replay=lambda source: logits)
    set_predictions([11, 12, 13, 14, 15])
    return SimpleNamespace(target=target, batch=batch, restored=restored, set_predictions=set_predictions)


@pytest.mark.parametrize("matched", range(5))
def test_every_target_acceptance_commits_exact_prefix_and_holds_ownership(target, matched):
    t = target
    tokens = [11, 12, 13, 14, 15]
    if matched < 4: tokens[matched] = 23
    t.set_predictions(tokens)
    output = t.target.forward(t.batch)
    assert output.tolist() == tokens[:matched + 1]
    assert t.batch.reqs[0].cached_len == 11 + matched
    assert t.batch.reqs[0].device_len == 12 + matched
    assert t.restored == ([matched + 1] if matched < 4 else [])
    assert t.target.owner is t.batch
    with pytest.raises(RuntimeError, match="owned"):
        t.target.forward(t.batch)
    with pytest.raises(RuntimeError, match="own"):
        t.target.release(object())
    t.target.release(t.batch)
    assert t.target.owner is None


@pytest.mark.parametrize("position", range(5))
def test_eos_or_tool_opener_limits_the_committed_window(target, position):
    target.batch.ngram_interrupt_ids = (11 + position,)
    output = target.target.forward(target.batch)
    assert output.tolist() == list(range(11, 12 + position))
    assert target.batch.generated_tokens == position + 1
    assert target.batch.reqs[0].cached_len == 11 + position


def test_host_stop_trims_full_acceptance_before_release(target):
    target.target.forward(target.batch)
    target.target.trim(target.batch, 2)
    assert target.restored == [2]
    assert target.batch.reqs[0].cached_len == 12
    assert target.batch.reqs[0].device_len == 13
    assert target.batch.generated_tokens == 2
    target.target.release(target.batch)
    with pytest.raises(RuntimeError, match="owned"):
        target.target.trim(target.batch, 1)


def test_failed_forward_releases_ownership(target):
    def fail(batch): raise RuntimeError("execution failure")
    target.target.graph.replay = fail
    with pytest.raises(RuntimeError, match="execution failure"):
        target.target.forward(target.batch)
    assert target.target.owner is None
