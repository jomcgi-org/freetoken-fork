"""Reject unsupported ngram ownership modes before any model load."""

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


def config(**changes):
    values = dict(model_path="unused", tp_info=DistributedInfo(rank=0, size=1),
                  dtype=torch.bfloat16, max_running_req=1)
    values.update(changes)
    return EngineConfig(**values)


def test_ngram_and_debug_default_off():
    assert config().speculative_ngram == "off"
    assert config().ngram_debug is False
    assert config(speculative_ngram="on", ngram_debug=True).ngram_debug


@pytest.mark.parametrize("changes", [dict(speculative_ngram="invalid"), dict(ngram_debug=True),
    dict(speculative_ngram="on", speculative_mtp="on"),
    dict(speculative_ngram="on", max_running_req=2),
    dict(speculative_ngram="on", tp_info=DistributedInfo(rank=0, size=2))])
def test_unsupported_configuration_rejected(changes):
    with pytest.raises(ValueError, match="ngram"):
        config(**changes)
