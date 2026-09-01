"""Optional XGrammar integration and protocol-neutral grammar specifications.

Importing this module does not import XGrammar. The native optional dependency is
loaded only when a request actually carries ``SamplingParams.guided_decoding``.
"""

from __future__ import annotations

import copy
import importlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch


class GuidedDecodingUnavailable(RuntimeError):
    pass


_TOOL_PARSER_STYLES = {
    "deepseekv32": "deepseek_v3_2",
    "glm47": "glm_4_7",
    "gpt-oss": "harmony",
    "gpt_oss": "harmony",
    "llama3": "llama",
    "minimax": "minimax",
    "qwen": "qwen_3",
    "qwen25": "qwen_3",
    "qwen3_coder": "qwen_3_coder",
}


def xgrammar_style_for_parser(parser: str, model_hint: str = "") -> str:
    if parser == "deepseekv32" and "v4" in model_hint.lower():
        return "deepseek_v4"
    try:
        return _TOOL_PARSER_STYLES[parser]
    except KeyError as exc:
        raise ValueError(
            f"tool_call_parser {parser!r} has no XGrammar structural grammar"
        ) from exc


def import_xgrammar() -> Any:
    try:
        return importlib.import_module("xgrammar")
    except (ImportError, OSError) as exc:
        raise GuidedDecodingUnavailable(
            "constrained decoding requires the optional dependency; "
            "install it with `pip install 'freetoken[guided]'`"
        ) from exc


def normalize_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert OpenAI response_format into an engine grammar specification."""
    if response_format is None or response_format.get("type") in (None, "text"):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return {"kind": "json_schema", "schema": {"type": "object"}, "strict": False}
    if kind != "json_schema":
        raise ValueError(
            "response_format.type must be 'text', 'json_object', or 'json_schema'"
        )
    envelope = response_format.get("json_schema")
    if not isinstance(envelope, dict):
        raise ValueError("response_format.json_schema must be an object")
    schema = envelope.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("response_format.json_schema.schema must be an object")
    return {
        "kind": "json_schema",
        "schema": schema,
        "strict": bool(envelope.get("strict", True)),
    }


def tool_constraint(
    tools: list[dict[str, Any]],
    *,
    tool_call_parser: str,
    model_hint: str = "",
    reasoning: bool,
    force_reasoning: bool,
) -> dict[str, Any]:
    if not tools:
        raise ValueError("tool_choice 'required' requires at least one tool")
    constrained_tools = copy.deepcopy(tools)
    for tool in constrained_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
            # Required mode is the strict contract shipped here: every declared
            # parameter schema is enforced even when the wire omitted `strict`.
            function["strict"] = True
    return {
        "kind": "tool",
        "style": xgrammar_style_for_parser(tool_call_parser, model_hint),
        "tools": constrained_tools,
        "tool_choice": "required",
        "reasoning": reasoning,
        "force_reasoning": force_reasoning,
    }


@dataclass
class GuidedState:
    matcher: Any
    start_after_ids: tuple[int, ...] = ()
    active: bool = True
    _trigger_prefix: list[int] = field(default_factory=list)

    @property
    def terminated(self) -> bool:
        return self.active and bool(self.matcher.is_terminated())

    def accept_token(self, token_id: int) -> None:
        if self.active:
            if not self.matcher.accept_token(token_id):
                raise RuntimeError(
                    f"guided decoding sampled token {token_id} rejected by its grammar"
                )
            return
        self._trigger_prefix.append(token_id)
        trigger = self.start_after_ids
        max_len = min(len(trigger), len(self._trigger_prefix))
        keep = 0
        for length in range(max_len, 0, -1):
            if tuple(self._trigger_prefix[-length:]) == trigger[:length]:
                keep = length
                break
        self._trigger_prefix = self._trigger_prefix[-keep:] if keep else []
        if keep == len(trigger):
            self.active = True
            self._trigger_prefix.clear()


@dataclass
class GuidedBatch:
    rows: list[int]
    states: list[GuidedState]
    cpu_us: float = 0.0
    gpu_started: torch.cuda.Event | None = None
    gpu_ended: torch.cuda.Event | None = None

    def elapsed_us(self) -> float:
        gpu_us = 0.0
        if self.gpu_started is not None and self.gpu_ended is not None:
            gpu_us = float(self.gpu_started.elapsed_time(self.gpu_ended) * 1000.0)
        return self.cpu_us + gpu_us


class XGrammarDecoder:
    """One per model. Compiler caching is shared by all request matchers."""

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        self.xgr = import_xgrammar()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        tokenizer_info = self.xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        self.compiler = self.xgr.GrammarCompiler(tokenizer_info)
        self.batch_matcher = self.xgr.BatchGrammarMatcher(max_threads="auto")

    def create_state(self, spec: dict[str, Any]) -> GuidedState:
        kind = spec.get("kind")
        if kind == "json_schema":
            compiled = self.compiler.compile_json_schema(
                spec["schema"], strict_mode=bool(spec.get("strict", True))
            )
        elif kind == "tool":
            structural_tag = self.xgr.get_model_structural_tag(
                spec["style"],
                tools=spec["tools"],
                tool_choice=spec.get("tool_choice", "required"),
                reasoning=bool(spec.get("reasoning", True)),
                force_reasoning=bool(spec.get("force_reasoning", False)),
            )
            compiled = self.compiler.compile_structural_tag(structural_tag)
        else:
            raise ValueError(f"unknown guided decoding kind {kind!r}")

        start_after = spec.get("start_after")
        trigger = tuple(
            self.tokenizer.encode(start_after, add_special_tokens=False)
            if start_after else []
        )
        if start_after and not trigger:
            raise ValueError(f"guided decoding trigger {start_after!r} tokenized to no ids")
        return GuidedState(
            matcher=self.xgr.GrammarMatcher(compiled),
            start_after_ids=trigger,
            active=not trigger,
        )

    def prepare(self, reqs: list[Any]) -> tuple[GuidedBatch | None, int]:
        rows: list[int] = []
        states: list[GuidedState] = []
        created = 0
        for row, req in enumerate(reqs):
            spec = getattr(req.sampling_params, "guided_decoding", None)
            # Intermediate ChunkedReq objects are forwarded but never emitted. Their final
            # continuation becomes an ordinary Req and receives the matcher exactly once.
            if spec is None or not req.can_decode:
                continue
            if req.guided_state is None:
                req.guided_state = self.create_state(spec)
                created += 1
            if req.guided_state.active:
                rows.append(row)
                states.append(req.guided_state)
        return (GuidedBatch(rows, states) if states else None), created

    def apply_mask(self, logits: torch.Tensor, guided: GuidedBatch) -> None:
        started = time.perf_counter_ns()
        bitmask = self.xgr.allocate_token_bitmask(logits.shape[0], self.vocab_size)
        self.batch_matcher.batch_fill_next_token_bitmask(
            [state.matcher for state in guided.states], bitmask, indices=guided.rows
        )
        guided.cpu_us += (time.perf_counter_ns() - started) / 1000.0

        apply_started = time.perf_counter_ns()
        if logits.is_cuda:
            guided.gpu_started = torch.cuda.Event(enable_timing=True)
            guided.gpu_ended = torch.cuda.Event(enable_timing=True)
            guided.gpu_started.record()
        device_mask = bitmask.to(logits.device, non_blocking=logits.is_cuda)
        self.xgr.apply_token_bitmask_inplace(
            logits, device_mask, vocab_size=self.vocab_size, indices=guided.rows
        )
        if logits.is_cuda:
            guided.gpu_ended.record()
        else:
            guided.cpu_us += (time.perf_counter_ns() - apply_started) / 1000.0

    def accept_tokens(self, guided: GuidedBatch, tokens: torch.Tensor) -> None:
        for row, state in zip(guided.rows, guided.states, strict=True):
            state.accept_token(int(tokens[row].item()))

    def observe_dormant(self, reqs: list[Any], tokens: torch.Tensor) -> None:
        """Advance delayed Qwen-style response grammars until their reasoning closer."""
        for row, req in enumerate(reqs):
            state = getattr(req, "guided_state", None)
            if state is not None and not state.active:
                state.accept_token(int(tokens[row].item()))
