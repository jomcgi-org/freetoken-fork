from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.guided as guided_module
from freetoken.guided import (
    GuidedDecodingUnavailable,
    GuidedState,
    XGrammarDecoder,
    import_xgrammar,
)
from freetoken.engine.sample import Sampler


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"</think>": [7, 8]}.get(text, [])


class _Matcher:
    def __init__(self, compiled):
        self.compiled = compiled
        self.accepted = []

    def accept_token(self, token):
        self.accepted.append(token)
        return token in self.compiled.allowed

    def is_terminated(self):
        return False


class _Compiler:
    def __init__(self, tokenizer_info):
        self.calls = []

    def compile_json_schema(self, schema, strict_mode=True):
        self.calls.append(("json_schema", schema, strict_mode))
        return SimpleNamespace(allowed={1, 3})

    def compile_structural_tag(self, tag):
        self.calls.append(("structural_tag", tag))
        return SimpleNamespace(allowed={2})


class _BatchMatcher:
    def __init__(self, max_threads="auto"):
        pass

    def batch_fill_next_token_bitmask(self, matchers, bitmask, indices=None):
        for matcher, row in zip(matchers, indices, strict=True):
            for token in matcher.compiled.allowed:
                bitmask[row, token // 32] |= 1 << (token % 32)


class _TokenizerInfo:
    @staticmethod
    def from_huggingface(tokenizer, vocab_size):
        return (tokenizer, vocab_size)


class _FakeXGrammar:
    TokenizerInfo = _TokenizerInfo
    GrammarCompiler = _Compiler
    BatchGrammarMatcher = _BatchMatcher
    GrammarMatcher = _Matcher

    def __init__(self):
        self.structural_calls = []

    @staticmethod
    def allocate_token_bitmask(batch_size, vocab_size):
        return torch.zeros((batch_size, (vocab_size + 31) // 32), dtype=torch.int32)

    @staticmethod
    def apply_token_bitmask_inplace(logits, bitmask, vocab_size=None, indices=None):
        for row in indices:
            allowed = torch.tensor(
                [
                    bool(int(bitmask[row, token // 32]) & (1 << (token % 32)))
                    for token in range(vocab_size)
                ]
            )
            logits[row].masked_fill_(~allowed, -torch.inf)

    def get_model_structural_tag(self, style, **kwargs):
        tag = {"style": style, **kwargs}
        self.structural_calls.append(tag)
        return tag


def _decoder(monkeypatch, vocab_size=4):
    fake = _FakeXGrammar()
    monkeypatch.setattr(guided_module, "import_xgrammar", lambda: fake)
    return XGrammarDecoder(_Tokenizer(), vocab_size), fake


def test_json_schema_compiles_to_request_matcher(monkeypatch):
    decoder, _ = _decoder(monkeypatch)
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    state = decoder.create_state(
        {"kind": "json_schema", "schema": schema, "strict": True}
    )

    assert isinstance(state.matcher, _Matcher)
    assert decoder.compiler.calls == [("json_schema", schema, True)]


def test_tool_schema_compiles_through_model_structural_tag(monkeypatch):
    decoder, fake = _decoder(monkeypatch)
    tools = [{
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {"type": "object"},
            "strict": True,
        },
    }]

    decoder.create_state({
        "kind": "tool",
        "style": "qwen_3_coder",
        "tools": tools,
        "tool_choice": "required",
        "reasoning": True,
        "force_reasoning": True,
    })

    assert fake.structural_calls == [{
        "style": "qwen_3_coder",
        "tools": tools,
        "tool_choice": "required",
        "reasoning": True,
        "force_reasoning": True,
    }]
    assert decoder.compiler.calls[0][0] == "structural_tag"


def test_mask_sets_invalid_logits_to_negative_infinity_in_mixed_batch(monkeypatch):
    decoder, _ = _decoder(monkeypatch)
    state = decoder.create_state(
        {"kind": "json_schema", "schema": {"type": "object"}}
    )
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(guided_decoding={"kind": "json_schema"}),
        guided_state=state,
        can_decode=True,
    )
    plain = SimpleNamespace(
        sampling_params=SimpleNamespace(guided_decoding=None),
        guided_state=None,
        can_decode=True,
    )
    batch, created = decoder.prepare([req, plain])
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]])

    decoder.apply_mask(logits, batch)

    assert created == 0
    assert logits[0].tolist() == [-torch.inf, 8.0, -torch.inf, 6.0]
    assert logits[1].tolist() == [5.0, 4.0, 3.0, 2.0]


def test_delayed_matcher_activates_only_after_complete_reasoning_closer():
    matcher = SimpleNamespace(accept_token=lambda token: True)
    state = GuidedState(matcher, start_after_ids=(7, 8), active=False)

    for token in (1, 7, 2, 7):
        state.accept_token(token)
        assert state.active is False
    state.accept_token(8)

    assert state.active is True


def test_optional_dependency_error_names_install_extra(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("xgrammar")

    monkeypatch.setattr(guided_module.importlib, "import_module", missing)
    with pytest.raises(GuidedDecodingUnavailable, match=r"freetoken\[guided\]"):
        import_xgrammar()


def test_unconstrained_sampler_prepare_does_not_initialize_backend(monkeypatch):
    def unexpected():
        raise AssertionError("guided backend should not initialize")

    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    monkeypatch.setattr(sampler, "_get_guided_decoder", unexpected)
    req = SimpleNamespace(
        can_decode=True,
        sampling_params=SimpleNamespace(
            guided_decoding=None,
            is_greedy=True,
        )
    )
    batch = SimpleNamespace(reqs=[req], constrained_requests=0)

    args = sampler.prepare(batch)

    assert args.guided is None
    assert args.has_guided is False
    assert batch.constrained_requests == 0


def test_real_xgrammar_schema_mask_when_optional_extra_is_installed():
    xgr = pytest.importorskip("xgrammar")
    # One raw token per ASCII byte plus an explicit stop token makes the expected
    # initial JSON-object mask independent of a production tokenizer vocabulary.
    vocab = [bytes([value]) for value in range(128)] + [b"<eos>"]
    info = xgr.TokenizerInfo(
        vocab, xgr.VocabType.RAW, vocab_size=len(vocab), stop_token_ids=[128]
    )
    compiled = xgr.GrammarCompiler(info).compile_json_schema({"type": "object"})
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, len(vocab))
    matcher.fill_next_token_bitmask(bitmask)
    logits = torch.zeros((1, len(vocab)))

    xgr.apply_token_bitmask_inplace(logits, bitmask, vocab_size=len(vocab))

    assert torch.isfinite(logits[0, ord("{")])
    assert torch.isneginf(logits[0, ord("x")])
