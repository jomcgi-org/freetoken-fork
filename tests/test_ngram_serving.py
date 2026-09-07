"""Causal drafts and target ownership preserve exact committed-prefix semantics."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch

from freetoken.verification.ngram import (
    propose, proposal_for_request, proposal_eligible, note_verification,
)
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


@pytest.mark.parametrize("last_token", [0, 7])
def test_pending_host_token_lookup_matches_committed_history_without_mutation(last_token):
    req = req_fixture()
    req.input_ids[7] = req.input_ids[-1] = last_token
    expected = proposal_for_request(req)
    req.input_ids = req.input_ids[:-1]
    history = req.input_ids.clone()
    before = dict(vars(req))
    assert proposal_eligible(req, pending_tokens=1)
    assert not proposal_eligible(req)
    assert proposal_for_request(req) is None
    assert proposal_for_request(req, pending_token=last_token) == expected == [31, 32, 33, 34]
    assert torch.equal(req.input_ids, history)
    assert all(vars(req)[key] is value for key, value in before.items())


def test_pending_lookup_requires_exactly_one_missing_host_token():
    req = req_fixture()
    assert not proposal_eligible(req, pending_tokens=1)
    req.input_ids = req.input_ids[:-2]
    assert proposal_for_request(req, pending_token=7) is None
    for count in (-1, 2):
        with pytest.raises(ValueError, match="pending token"):
            proposal_eligible(req, pending_tokens=count)


def test_weak_proposal_pauses_until_exact_retry_position_without_mutating_history():
    req = req_fixture()
    history = req.input_ids.clone()
    note_verification(req, 0)
    assert torch.equal(req.input_ids, history)
    assert req._ngram_retry_at == 36
    for length in (20, 35, 36):
        req.input_ids = torch.cat((torch.full((length - 20,), 99), history))
        req.cached_len, req.device_len = length - 1, length
        actual = proposal_for_request(req)
        assert actual == ([31, 32, 33, 34] if length == 36 else None)


def test_repeated_weak_proposals_back_off_with_a_bounded_delay():
    req = req_fixture()
    for delay in (16, 32, 64, 128, 256, 256, 256):
        note_verification(req, 1)
        assert req._ngram_retry_at - req.device_len == delay
        req.device_len = req._ngram_retry_at + 2


@pytest.mark.parametrize("matched", [2, 3, 4])
def test_productive_proposal_resets_only_its_requests_backoff(matched):
    first, second = req_fixture(), req_fixture()
    note_verification(first, 0)
    assert proposal_for_request(first) is None
    assert proposal_for_request(second) == [31, 32, 33, 34]
    note_verification(first, matched)
    assert first._ngram_weak_windows == 0
    assert proposal_for_request(first) == [31, 32, 33, 34]
    note_verification(first, 0)
    assert first._ngram_retry_at - first.device_len == 16


@pytest.mark.parametrize("matched", [-1, 5])
def test_invalid_acceptance_does_not_change_request_backoff(matched):
    req = req_fixture()
    before = dict(vars(req))
    with pytest.raises(ValueError, match="acceptance"):
        note_verification(req, matched)
    assert vars(req) == before


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
    assert t.batch.reqs[0]._ngram_retry_at == (28 + matched if matched < 2 else 0)
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
    assert target.batch.reqs[0]._ngram_retry_at == 0


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


@pytest.mark.parametrize("failure", [False, True])
def test_startup_capture_matches_scheduler_index_types_and_restores_padding(monkeypatch, failure):
    monkeypatch.setitem(sys.modules, "freetoken.attention.linear",
                        SimpleNamespace(build_fla_metadata=lambda batch, device: None))
    pool = SimpleNamespace(conv_states=torch.ones(1, 2, 2), recurrent_states=torch.ones(1, 2, 3),
                            slot_states={"ple_conv": torch.ones(1, 2, 2),
                                         "ple_ngram_ctx": torch.ones(1, 2, 8, dtype=torch.int64)},
                            _state_layer_index={"ple_conv": {2: 0}})
    kv = SimpleNamespace(index_ratio=4, cmp_scratch_base=6,
                         _kv_buffer=torch.ones(1, 2, 3, 8, 1, 2),
                         _cmp_k_buffer=torch.ones(1, 8, 2),
                         _pending_ring=torch.ones(2, 1, 8, 2))
    engine = SimpleNamespace(device="cpu", config=SimpleNamespace(page_size=8, ngram_debug=False),
                             num_pages=2, page_table=torch.zeros(2, 16, dtype=torch.int32),
                             linear_state_pool=pool, kv_cache=kv,
                             model=SimpleNamespace(_ple_disk_decode=[object()]),
                             cpu_moe_executor=SimpleNamespace(quant_format="nvfp4"),
                             graph_runner=SimpleNamespace(graph_map={1: object()}),
                             stream=SimpleNamespace(synchronize=lambda: None),
                             attn_backend=SimpleNamespace(_idx_slot={7: 0}, prepare_metadata=lambda batch: None),
                             dummy_req=SimpleNamespace(table_idx=1, linear_slot_idx=0))
    buffers = [engine.page_table, kv._kv_buffer, kv._cmp_k_buffer, kv._pending_ring,
               pool.conv_states, pool.recurrent_states, *pool.slot_states.values()]
    before = [value.clone() for value in buffers]

    def capture(engine, batch, *, state_checkpoint):
        assert batch.active_table_idx.dtype == torch.int64
        assert batch.active_table_idx.tolist() == [1] * 5
        assert batch.linear_table_idx.dtype == torch.int32
        assert batch.out_loc.tolist() == [16, 17, 18, 19, 20]
        assert batch.positions.tolist() == list(range(5))
        # Simulate exactly the storage a dummy target is allowed to mutate.
        engine.page_table[1, :8].fill_(99)
        kv._kv_buffer[:, :, 2].fill_(99)
        kv._cmp_k_buffer[:, 4:6].fill_(99)
        kv._cmp_k_buffer[:, 7].fill_(99)
        for value in runtime.state_views(engine, batch.reqs[0]).values():
            value.fill_(99)
        if failure:
            raise RuntimeError("capture failure")
        return SimpleNamespace()

    monkeypatch.setattr(runtime.adapters, "FusedGraph", capture)
    target = NgramTarget(engine)
    if failure:
        with pytest.raises(RuntimeError, match="capture failure"):
            target.initialize()
    else:
        target.initialize()
    assert all(torch.equal(value, prior) for value, prior in zip(buffers, before))
