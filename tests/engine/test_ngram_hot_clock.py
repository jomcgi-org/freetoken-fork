"""HOT adaptation counts executed target rows, including rejected candidates."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def forward(monkeypatch):
    from freetoken.engine.engine import Engine

    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None))

    def run(*, ngram=False, accepted=1, prefill=False, fail=False):
        calls = []
        reqs = [SimpleNamespace(complete_one=lambda: calls.append("complete"))
                for _ in range(1 if ngram else 2)]
        batch = SimpleNamespace(
            is_decode=not prefill, is_prefill=prefill, size=len(reqs), reqs=reqs,
            input_ids=torch.zeros(5 if ngram else 20 if prefill else 4, dtype=torch.int32),
            ngram_verify=ngram, mtp_verify=ngram,
        )

        def target(batch):
            calls.append("target")
            if fail:
                raise RuntimeError("forward failed")
            batch.generated_tokens = accepted
            batch.mtp_fused = False
            return torch.arange(accepted, dtype=torch.int32)

        def ordinary(*_):
            calls.append("ordinary")
            if fail:
                raise RuntimeError("forward failed")
            return torch.zeros(4, 8)

        engine = SimpleNamespace(
            stream=stream, cpu_moe_executor=None,
            config=SimpleNamespace(moe_step_timing=False, speculative_mtp="off"),
            ctx=SimpleNamespace(forward_batch=lambda _: nullcontext()),
            ngram_target=SimpleNamespace(forward=target),
            graph_runner=SimpleNamespace(can_use_cuda_graph=lambda b: b.is_decode, replay=ordinary),
            model=SimpleNamespace(forward=ordinary),
            sampler=SimpleNamespace(sample=lambda logits, _: torch.argmax(logits, dim=-1)),
            _record_mtp_hidden=lambda _: None,
            moe_offload_cache=SimpleNamespace(
                hot_adapt_step_boundary=lambda count: calls.append(("decode", count)),
                hot_adapt_prefill_boundary=lambda: calls.append("prefill"),
            ),
        )
        args = SimpleNamespace(has_guided=False)
        if fail:
            with pytest.raises(RuntimeError, match="forward failed"):
                Engine.forward_batch(engine, batch, args)
            return calls, None
        return calls, Engine.forward_batch(engine, batch, args)

    return run


@pytest.mark.parametrize("accepted", [1, 3, 5])
def test_all_evaluated_ngram_rows_advance_clock_even_when_drafts_are_rejected(forward, accepted):
    calls, output = forward(ngram=True, accepted=accepted)
    assert calls == ["target", ("decode", 5)]
    assert output.next_tokens_cpu.numel() == accepted


def test_ordinary_padding_does_not_count_as_routed_requests(forward):
    calls, output = forward()
    assert calls == ["ordinary", ("decode", 2), "complete", "complete"]
    assert output.next_tokens_cpu.numel() == 2


def test_prefill_keeps_its_existing_token_accounting_boundary(forward):
    calls, _ = forward(prefill=True)
    assert calls == ["ordinary", "prefill", "complete", "complete"]


@pytest.mark.parametrize("ngram", [False, True])
def test_failed_forward_does_not_advance_hot_clock(forward, ngram):
    calls, _ = forward(ngram=ngram, fail=True)
    assert calls == ["target" if ngram else "ordinary"]
