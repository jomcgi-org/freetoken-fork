"""Hard reasoning-token budget: force the end-of-thinking tag once N tokens are spent.

Torch-free so the forcing logic is unit-testable without a GPU. The sampler owns one
``ReasoningBudgetState`` per budgeted request, feeds it every sampled token on the host,
and asks it which token (if any) the next step must emit. Forcing is expressed as a
single-token logit mask applied outside the CUDA graph, only on rows whose budget is spent.

With ``budget == 0`` the same state is a pure phase tracker (never forces a token): the
scheduler feeds it the tokens it already reads on the host, and the HOT expert adapter
asks ``reasoning_active`` to pick which decode history a step belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReasoningBudgetState:
    budget: int
    end_ids: tuple[int, ...]
    start_ids: tuple[int, ...] = ()
    in_reasoning: bool = True
    done: bool = False
    count: int = 0
    _tail: list[int] = field(default_factory=list)

    @property
    def forced_token(self) -> int | None:
        """The token the next step must emit, or None while sampling is unconstrained.

        A multi-token end tag is forced one token at a time; a prefix the model already
        emitted on its own is continued rather than restarted."""
        if self.budget <= 0 or self.done or not self.in_reasoning or self.count < self.budget:
            return None
        return self.end_ids[self._match(self.end_ids)]

    @property
    def reasoning_active(self) -> bool:
        """True while the stream is inside an open reasoning block (``done`` stays set
        after the end tag, so ``in_reasoning`` alone would stay true)."""
        return self.in_reasoning and not self.done

    def observe(self, token_id: int) -> None:
        """Advance by one sampled token (forced or free)."""
        if self.done:
            return
        self._tail.append(token_id)
        keep = max(len(self.end_ids), len(self.start_ids))
        if len(self._tail) > keep:
            del self._tail[: len(self._tail) - keep]
        if not self.in_reasoning:
            if self.start_ids and self._match(self.start_ids) == len(self.start_ids):
                self.in_reasoning = True
                self._tail.clear()
            return
        if self._match(self.end_ids) == len(self.end_ids):
            self.done = True
            return
        self.count += 1

    def _match(self, seq: tuple[int, ...]) -> int:
        """Length of the longest suffix of the emitted tail that is a prefix of ``seq``."""
        for length in range(min(len(seq), len(self._tail)), 0, -1):
            if tuple(self._tail[-length:]) == seq[:length]:
                return length
        return 0


def create_reasoning_budget_state(spec: dict, tokenizer) -> ReasoningBudgetState:
    """Build per-request state from ``SamplingParams.reasoning_budget``
    (``tokens``, ``end``, optional ``start``, ``open``). ``SamplingParams.reasoning_phase``
    uses the same spec without ``tokens``, giving a non-forcing tracker."""
    end_ids = tuple(tokenizer.encode(spec["end"], add_special_tokens=False))
    if not end_ids:
        raise ValueError(f"reasoning end tag {spec['end']!r} tokenized to no ids")
    start = spec.get("start")
    start_ids = tuple(tokenizer.encode(start, add_special_tokens=False)) if start else ()
    in_reasoning = bool(spec.get("open", True)) or not start_ids
    return ReasoningBudgetState(
        budget=int(spec.get("tokens", 0)),
        end_ids=end_ids,
        start_ids=start_ids,
        in_reasoning=in_reasoning,
    )
