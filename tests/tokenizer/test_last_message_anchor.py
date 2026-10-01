from __future__ import annotations

import re

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager

_BASE = 1_000_000


class MarkupTokenizer:
    """Character tokenizer that splits registered markup strings out as atomic tokens,
    the way an HF tokenizer treats special and added tokens."""

    def __init__(self, markup, template, special=None):
        self._markup = {text: _BASE + i for i, text in enumerate(markup)}
        self._special = set(markup if special is None else special)
        self._template = template
        self._split = re.compile("(" + "|".join(map(re.escape, markup)) + ")")

    @property
    def all_special_ids(self):
        return [self._markup[text] for text in self._special]

    def get_added_vocab(self):
        return dict(self._markup)

    def apply_chat_template(self, messages, **kwargs):
        return self._template(messages, kwargs.get("tools"))

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        ids = []
        for piece in self._split.split(prompt):
            if piece in self._markup:
                ids.append(self._markup[piece])
            else:
                ids.extend(ord(char) for char in piece)
        return torch.tensor([ids], dtype=torch.long)


def _text(content):
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content or ""


def _chatml(messages, tools):
    out = f"<|im_start|>system\n[tools {tools}]<|im_end|>\n" if tools else ""
    for i, message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += f"\n<tool_response>\n{_text(message['content'])}\n</tool_response>"
            if i + 1 == len(messages) or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
            continue
        calls = "".join(
            f"\n<tool_call>\n{call}\n</tool_call>" for call in message.get("tool_calls") or []
        )
        out += f"<|im_start|>{role}\n{_text(message.get('content'))}{calls}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def _llama(messages, tools):
    out = "<|begin_of_text|>"
    for message in messages:
        out += (
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{_text(message['content'])}<|eot_id|>"
        )
    return out + "<|start_header_id|>assistant<|end_header_id|>\n\n"


CHATML_MARKUP = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
                 "<tool_response>", "</tool_response>"]
LLAMA_MARKUP = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def _chatml_manager(enabled=True):
    # Only the im_* tokens are "special"; the rest are merely added, as in Qwen.
    tokenizer = MarkupTokenizer(CHATML_MARKUP, _chatml, special=CHATML_MARKUP[:2])
    return TokenizeManager(tokenizer, harness_prefixes=(), last_message_anchor=enabled)


def _llama_manager():
    tokenizer = MarkupTokenizer(LLAMA_MARKUP, _llama)
    return TokenizeManager(tokenizer, harness_prefixes=(), last_message_anchor=True)


def _anchor(manager, messages, tools=None):
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams(), tools=tools)
    ids, _, _ = manager.tokenize_with_cache_anchor(msg)
    return ids, manager.last_message_anchor(msg, ids)


HISTORY = [
    {"role": "system", "content": "You are an agent. " * 5},
    {"role": "user", "content": "first task"},
    {"role": "assistant", "content": "ok", "tool_calls": ['{"name":"ls"}']},
    {"role": "tool", "content": "a.py b.py"},
    {"role": "assistant", "content": "listed"},
]
TOOLS = [{"type": "function", "function": {"name": "ls"}}]


@pytest.mark.parametrize("make", [_chatml_manager, _llama_manager])
def test_anchor_is_stable_across_different_last_user_messages(make):
    manager = make()
    ids_a, anchor_a = _anchor(manager, [*HISTORY, {"role": "user", "content": "short"}], TOOLS)
    ids_b, anchor_b = _anchor(
        manager, [*HISTORY, {"role": "user", "content": "a very different, longer follow-up"}], TOOLS
    )
    assert anchor_a is not None and anchor_a == anchor_b
    assert torch.equal(ids_a[:anchor_a], ids_b[:anchor_a])
    assert anchor_a < ids_a.numel()


@pytest.mark.parametrize("make", [_chatml_manager, _llama_manager])
def test_anchor_opens_the_last_message_turn(make):
    manager = make()
    ids, anchor = _anchor(manager, [*HISTORY, {"role": "user", "content": "go"}])
    # the token before the anchor is turn markup; what follows is the role/content
    assert int(ids[anchor - 1]) >= _BASE
    tail = bytes(int(i) for i in ids[anchor:] if int(i) < _BASE).decode()
    assert tail.startswith("user" if make is _chatml_manager else "")
    assert "go" in tail


def test_anchor_is_deeper_than_the_system_prompt_and_covers_history():
    manager = _chatml_manager()
    short_ids, short = _anchor(manager, [HISTORY[0], {"role": "user", "content": "x"}])
    long_ids, long = _anchor(manager, [*HISTORY, {"role": "user", "content": "x"}])
    assert long > short
    assert long_ids.numel() > long


def test_tool_result_last_message_anchors_after_its_opener():
    manager = _chatml_manager()
    messages = HISTORY[:3]
    ids_a, anchor_a = _anchor(manager, [*messages, {"role": "tool", "content": "output one"}])
    ids_b, anchor_b = _anchor(
        manager, [*messages, {"role": "tool", "content": "completely different output"}]
    )
    assert anchor_a == anchor_b
    assert torch.equal(ids_a[:anchor_a], ids_b[:anchor_a])
    assert int(ids_a[anchor_a - 1]) == _BASE + CHATML_MARKUP.index("<tool_response>")


def test_second_consecutive_tool_result_opens_at_its_own_marker():
    manager = _chatml_manager()
    base = [*HISTORY[:3], {"role": "tool", "content": "one"}]
    _, one = _anchor(manager, base)
    _, two = _anchor(manager, [*base, {"role": "tool", "content": "two"}])
    assert two > one


def test_assistant_prefill_last_message():
    manager = _chatml_manager()
    _, anchor_a = _anchor(manager, [*HISTORY, {"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "Sure, "}])
    _, anchor_b = _anchor(manager, [*HISTORY, {"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "Certainly, here"}])
    assert anchor_a == anchor_b and anchor_a is not None


def test_content_parts_and_empty_assistant_content_with_tool_calls():
    manager = _chatml_manager()
    parts = [{"type": "text", "text": "hello"}]
    _, a = _anchor(manager, [*HISTORY, {"role": "user", "content": parts}])
    _, b = _anchor(manager, [*HISTORY, {"role": "user", "content": "other"}])
    assert a == b
    ids, anchor = _anchor(
        manager,
        [HISTORY[1], {"role": "assistant", "content": None, "tool_calls": ['{"n":1}']}],
    )
    assert anchor is not None and anchor < ids.numel()


def test_content_sharing_the_probe_start_still_anchors_at_the_turn():
    manager = _chatml_manager()
    _, plain = _anchor(manager, [*HISTORY, {"role": "user", "content": "x"}])
    _, odd = _anchor(manager, [*HISTORY, {"role": "user", "content": "probe tail"}])
    assert plain == odd


def test_disabled_flag_non_chat_and_unprobeable_templates_return_none():
    msgs = [*HISTORY, {"role": "user", "content": "x"}]
    assert _anchor(_chatml_manager(enabled=False), msgs)[1] is None
    manager = _chatml_manager()
    raw = TokenizeMsg(uid=1, text="raw prompt", sampling_params=SamplingParams())
    assert manager.last_message_anchor(raw, torch.tensor([1, 2, 3])) is None

    def ignores_content(messages, tools):
        return "<|im_start|>fixed<|im_end|>"

    ignoring = TokenizeManager(
        MarkupTokenizer(CHATML_MARKUP, ignores_content), harness_prefixes=(),
        last_message_anchor=True,
    )
    assert _anchor(ignoring, msgs)[1] is None

    def broken(messages, tools):
        if messages[-1]["content"].startswith(""):
            raise ValueError("template rejects the probe")
        return _chatml(messages, tools)

    rejecting = TokenizeManager(
        MarkupTokenizer(CHATML_MARKUP, broken), harness_prefixes=(), last_message_anchor=True
    )
    assert _anchor(rejecting, msgs)[1] is None


def test_prompt_without_any_markup_has_no_anchor():
    manager = TokenizeManager(
        MarkupTokenizer(CHATML_MARKUP, lambda m, t: "".join(_text(x["content"]) for x in m)),
        harness_prefixes=(), last_message_anchor=True,
    )
    assert _anchor(manager, [{"role": "user", "content": "plain"}])[1] is None
