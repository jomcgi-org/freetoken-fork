from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

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

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        guided, created, has_guided = self._prepare_guided(batch)
        batch.constrained_requests = created
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(
                temperatures=None, guided=guided, has_guided=has_guided
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
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.guided is not None:
                assert self._guided_decoder is not None
                self._guided_decoder.apply_mask(logits, args.guided)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

    def finish_guided(
        self, batch: Batch, args: BatchSamplingArgs, next_tokens_cpu: torch.Tensor
    ) -> float:
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
