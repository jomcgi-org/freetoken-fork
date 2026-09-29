"""Memory-aware expert batches for layer-major prefill."""

from types import SimpleNamespace

import pytest


def _mlp():
    return SimpleNamespace(experts=SimpleNamespace(top_k=10, intermediate_size=640))


def _prepared(tokens):
    return (None, tokens, 2560, None, None, None)


@pytest.mark.parametrize(
    ("free_gib", "expected"),
    [
        (8.0, [[4096, 4096, 4096, 4096]]),
        (2.0, [[4096, 4096], [4096, 4096]]),
        (1.5, [[4096], [4096], [4096], [4096]]),
        (0.1, [[4096], [4096], [4096], [4096]]),
    ],
)
def test_expert_batches_split_to_fit_free_memory(monkeypatch, free_gib, expected):
    import torch

    from freetoken.models.qwen4_exp import model as module

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(free_gib * 2**30), 24 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    parts = module._fit_expert_batch(_mlp(), [_prepared(4096) for _ in range(4)])
    assert [[p[1] for p in part] for part in parts] == expected


def test_single_chunk_is_never_split(monkeypatch):
    from freetoken.models.qwen4_exp import model as module

    prepared = [_prepared(4096)]
    assert module._fit_expert_batch(_mlp(), prepared) == [prepared]
