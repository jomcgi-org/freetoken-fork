from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, NamedTuple

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.guided import GuidedBatch, XGrammarDecoder


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    guided: "GuidedBatch | None" = None
    has_guided: bool = False
    # Reasoning budget: rows observed this step (host token ids needed afterwards) and the
    # (row, token) pairs whose logits are collapsed to the forced end-tag token.
    budget_reqs: "list[tuple[int, Any]]" = field(default_factory=list)
    forced: "list[tuple[int, int]]" = field(default_factory=list)

    @property
    def needs_host_tokens(self) -> bool:
        return self.has_guided or bool(self.budget_reqs)


class LogprobRows(NamedTuple):
    """Per-row logprobs for one sampled batch, on the host once the copy event fires.

    ``chosen[i]`` is the sampled token's logprob; ``top_ids``/``top_logprobs`` hold the
    batch-wide maximum requested alternatives, which each request slices to its own count.
    """

    chosen: torch.Tensor  # [bs] float32
    top_ids: torch.Tensor  # [bs, k] int32
    top_logprobs: torch.Tensor  # [bs, k] float32


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def force_tokens(logits: torch.Tensor, forced: list[tuple[int, int]]) -> None:
    """Collapse each (row, token) row of ``logits`` to that single token, in place."""
    rows = torch.tensor([r for r, _ in forced], dtype=torch.long, device=logits.device)
    ids = torch.tensor([t for _, t in forced], dtype=torch.long, device=logits.device)
    kept = logits[rows, ids]
    logits[rows] = float("-inf")
    logits[rows, ids] = kept


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def __post_init__(self) -> None:
        self._guided_tokenizer: Any | None = None
        self._guided_decoder: "XGrammarDecoder | None" = None

    def set_guided_tokenizer(self, tokenizer: Any) -> None:
        # Store only. XGrammar remains completely unloaded until a constrained request.
        self._guided_tokenizer = tokenizer

    def _get_guided_decoder(self):
        if self._guided_decoder is None:
            if self._guided_tokenizer is None:
                raise RuntimeError("guided decoding tokenizer was not initialized")
            from freetoken.guided import XGrammarDecoder

            self._guided_decoder = XGrammarDecoder(self._guided_tokenizer, self.vocab_size)
        return self._guided_decoder

    def validate_guided(self, spec: dict[str, Any]) -> None:
        # Compiles into the persistent compiler cache before admission. A bad client
        # schema becomes a request error instead of killing the scheduler during forward.
        self._get_guided_decoder().create_state(spec)

    def _prepare_guided(self, batch: Batch):
        has_guided = any(
            r.can_decode and r.sampling_params.guided_decoding is not None
            for r in batch.reqs
        )
        if not has_guided:
            return None, 0, False
        guided, created = self._get_guided_decoder().prepare(batch.reqs)
        return guided, created, True

    def _prepare_reasoning_budget(self, batch: Batch):
        budget_reqs: list[tuple[int, Any]] = []
        forced: list[tuple[int, int]] = []
        for row, req in enumerate(batch.reqs):
            spec = req.sampling_params.reasoning_budget
            if spec is None or not spec.get("tokens") or not req.can_decode:
                continue
            if req.reasoning_budget_state is None:
                from freetoken.reasoning_budget import create_reasoning_budget_state

                if self._guided_tokenizer is None:
                    raise RuntimeError("reasoning budget tokenizer was not initialized")
                req.reasoning_budget_state = create_reasoning_budget_state(
                    spec, self._guided_tokenizer
                )
            state = req.reasoning_budget_state
            if state.done:
                continue
            budget_reqs.append((row, state))
            token = state.forced_token
            # A live grammar owns this row's mask; forcing a token it may reject would
            # crash the matcher, so grammar-constrained rows are not budgeted.
            guided_state = req.guided_state
            if token is not None and not (guided_state is not None and guided_state.active):
                forced.append((row, token))
        return budget_reqs, forced

    def reasoning_phase_state(self, req: Any):
        """The request's non-forcing reasoning tracker, created on first use, or None
        when the request carries no tag spec (histories != split3, or no tag pair)."""
        state = req.reasoning_phase_state
        if state is None:
            spec = req.sampling_params.reasoning_phase
            if spec is None or self._guided_tokenizer is None:
                return None
            from freetoken.reasoning_budget import create_reasoning_budget_state

            state = req.reasoning_phase_state = create_reasoning_budget_state(
                spec, self._guided_tokenizer
            )
        return state

    def batch_in_reasoning(self, batch: Batch) -> bool:
        """Whether any row of ``batch`` is inside a reasoning block. The expert counters
        are per layer, not per row, so a mixed batch is attributed to reasoning if any
        row reasons (batch size is 1 in production; this keeps bs>1 well defined)."""
        reasoning = False
        for req in batch.reqs:
            state = self.reasoning_phase_state(req)
            if state is not None and state.reasoning_active:
                reasoning = True
        return reasoning

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        guided, created, has_guided = self._prepare_guided(batch)
        batch.constrained_requests = created
        budget_reqs, forced = (
            self._prepare_reasoning_budget(batch)
            if any(p.reasoning_budget is not None for p in params)
            else ([], [])
        )
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(
                temperatures=None, guided=guided, has_guided=has_guided,
                budget_reqs=budget_reqs, forced=forced,
            )

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperatures, top_k=top_k, top_p=top_p,
            guided=guided, has_guided=has_guided,
            budget_reqs=budget_reqs, forced=forced,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.guided is not None:
                assert self._guided_decoder is not None
                self._guided_decoder.apply_mask(logits, args.guided)
            if args.forced:
                force_tokens(logits, args.forced)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

    def logprobs(
        self, logits: torch.Tensor, next_tokens: torch.Tensor, batch: Batch
    ) -> LogprobRows | None:
        """Host copies of the model's log-softmax at the sampled positions, or None.

        Taken from the untempered logits the sampler saw, so a grammar-constrained row
        reports logprobs after its mask. Only batches with a logprobs request pay for it.
        """
        k = max((r.sampling_params.logprobs for r in batch.reqs), default=0)
        if k <= 0:
            return None
        rows = next_tokens.numel()
        logp = torch.log_softmax(logits[:rows, : self.vocab_size].float(), dim=-1)
        chosen = logp.gather(1, next_tokens.long().view(-1, 1)).view(-1)
        top_logprobs, top_ids = logp.topk(k, dim=-1)
        return LogprobRows(
            chosen.to("cpu", non_blocking=True),
            top_ids.to(torch.int32).to("cpu", non_blocking=True),
            top_logprobs.to("cpu", non_blocking=True),
        )

    def finish_guided(
        self, batch: Batch, args: BatchSamplingArgs, next_tokens_cpu: torch.Tensor
    ) -> float:
        for row, state in args.budget_reqs:
            state.observe(int(next_tokens_cpu[row].item()))
        if args.guided is None:
            # A delayed response grammar can have no active rows yet. It still must
            # observe unrestricted reasoning tokens to find its activation marker.
            if self._guided_decoder is not None:
                self._guided_decoder.observe_dormant(batch.reqs, next_tokens_cpu)
            return 0.0
        assert self._guided_decoder is not None
        self._guided_decoder.accept_tokens(args.guided, next_tokens_cpu)
        self._guided_decoder.observe_dormant(batch.reqs, next_tokens_cpu)
        return args.guided.elapsed_us()
