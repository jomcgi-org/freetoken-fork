from __future__ import annotations

import importlib.util
import json
import os
import threading
from collections import OrderedDict
from types import ModuleType
from typing import Any, List, Sequence

import torch
from freetoken.message import TokenizeMsg
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)

logger = init_logger(__name__)


_DEFAULT_HARNESS_PREFIXES = (
    "opencode=You are OpenCode,",
    "pi=You are a focused coding agent.",
)
_PREAMBLE_CACHE_SIZE = 64


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


class TokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        harness_prefixes: Sequence[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._harness_prefixes = _parse_harness_prefixes(
            _DEFAULT_HARNESS_PREFIXES if harness_prefixes is None else harness_prefixes
        )
        self._preamble_cache: OrderedDict[str, tuple[str, torch.Tensor]] = OrderedDict()
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        # TODO: batch tokenization
        return [
            self._encode_prompt(
                self.render_prompt(msg), templated=isinstance(msg.text, list)
            )
            for msg in msgs
        ]

    def tokenize_with_cache_anchor(
        self, msg: TokenizeMsg
    ) -> tuple[torch.Tensor, int | None, str | None]:
        """Tokenize one request and find a reusable coding-harness preamble.

        Configured coding harnesses put stable instructions and tool schemas before
        the first conversational message. Rendering that leading system run by itself,
        then taking its token-level common prefix with the real prompt, finds the
        exact template-safe boundary without assuming how a model spells role markers.
        Any generation suffix emitted by the standalone template stops contributing
        at the first token where it differs from the full conversation render.
        """
        prompt = self.render_prompt(msg)
        input_ids = self._encode_prompt(prompt, templated=isinstance(msg.text, list))
        kind, preamble = _known_harness_preamble(msg.text, self._harness_prefixes)
        if kind is None or not preamble:
            return input_ids, None, None
        try:
            _, preamble_ids = self._cached_preamble(
                preamble,
                msg.tools,
                self._sanitize_effort(msg.chat_template_kwargs or {}),
            )
        except Exception:  # a template may reject a system-only conversation
            return input_ids, None, None
        anchor = _common_prefix_len(input_ids, preamble_ids)
        if anchor <= 0 or anchor >= input_ids.numel():
            return input_ids, None, None
        return input_ids, anchor, kind

    def _cached_preamble(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> tuple[str, torch.Tensor]:
        key = json.dumps(
            (messages, tools, chat_template_kwargs),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=repr,
        )
        cached = self._preamble_cache.get(key)
        if cached is not None:
            self._preamble_cache.move_to_end(key)
            return cached
        rendered_preamble = self._render(messages, tools, chat_template_kwargs)
        encoded = self._encode_prompt(rendered_preamble, templated=True)
        cached = (rendered_preamble, encoded)
        self._preamble_cache[key] = cached
        self._preamble_cache.move_to_end(key)
        if len(self._preamble_cache) > _PREAMBLE_CACHE_SIZE:
            self._preamble_cache.popitem(last=False)
        return cached

    def _encode_prompt(self, prompt: str, *, templated: bool) -> torch.Tensor:
        # A jinja chat template owns every special token (HF's apply_chat_template
        # tokenizes with add_special_tokens=False for the same reason): tokenizers
        # that auto-add bos (muse-glimmer's, llama's) would otherwise double it.
        # Raw-string prompts and the dsv4 encoder path keep the default.
        rendered_by_jinja = templated and self._dsv4_encoder is None
        input_ids: torch.Tensor = self.tokenizer.encode(  # type: ignore
            prompt, return_tensors="pt", add_special_tokens=not rendered_by_jinja
        )
        return input_ids.view(-1).to(torch.int32)

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        return self._render(
            msg.text, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _known_harness_preamble(
    text: str | list[dict[str, Any]],
    prefixes: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    if not isinstance(text, list):
        return None, []
    preamble: list[dict[str, Any]] = []
    for message in text:
        if message.get("role") not in ("system", "developer"):
            break
        preamble.append(message)
    if not preamble:
        return None, []
    configured = (
        _parse_harness_prefixes(_DEFAULT_HARNESS_PREFIXES)
        if prefixes is None
        else prefixes
    )
    for message in preamble:
        system_text = _content_text(message.get("content")).lstrip().casefold()
        for kind, prefix in configured:
            if system_text.startswith(prefix):
                return kind, preamble
    return None, []


def _parse_harness_prefixes(entries: Sequence[str]) -> tuple[tuple[str, str], ...]:
    parsed = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(
                "harness prefixes must be strings using kind=prefix syntax, "
                f"got {entry!r}"
            )
        kind, separator, prefix = entry.partition("=")
        if not separator or not kind.strip() or not prefix.strip():
            raise ValueError(
                "harness prefixes must use non-empty kind=prefix syntax, "
                f"got {entry!r}"
            )
        parsed.append((kind.strip(), prefix.lstrip().casefold()))
    return tuple(parsed)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _common_prefix_len(left: torch.Tensor, right: torch.Tensor) -> int:
    limit = min(int(left.numel()), int(right.numel()))
    if limit == 0:
        return 0
    different = torch.nonzero(left[:limit] != right[:limit], as_tuple=False)
    return limit if different.numel() == 0 else int(different[0].item())


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
