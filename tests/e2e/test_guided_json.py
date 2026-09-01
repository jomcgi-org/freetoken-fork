"""CUDA-gated structured generation against a real local test checkpoint.

Set FREETOKEN_GUIDED_TEST_MODEL to a small local checkpoint and install
``freetoken[guided]``. The test drives the production engine and CUDA-graph decode path.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.guided import normalize_response_format
from freetoken.llm import LLM

pytestmark = [pytest.mark.cuda, pytest.mark.needs_weights]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="guided e2e needs CUDA")
def test_json_object_generation_is_valid_json():
    if importlib.util.find_spec("xgrammar") is None:
        pytest.skip("install freetoken[guided]")
    value = os.environ.get("FREETOKEN_GUIDED_TEST_MODEL")
    if not value:
        pytest.skip("set FREETOKEN_GUIDED_TEST_MODEL to a small local model directory")
    model_path = Path(value).expanduser()
    if not model_path.is_dir():
        pytest.skip(f"model is not downloaded: {model_path}")

    llm = LLM(
        model_path=str(model_path),
        attention_backend="auto",
        max_running_req=1,
        cuda_graph_max_bs=1,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        guided_decoding=normalize_response_format({"type": "json_object"}),
    )
    result = llm.generate(
        ['Return a JSON object with the string field "status" set to "ok".\nJSON:'],
        sampling,
    )[0]

    parsed = json.loads(str(result["text"]))
    assert isinstance(parsed, dict)
