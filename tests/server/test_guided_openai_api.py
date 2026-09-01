from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.openai_api import chat_request_to_genspec


def _request(**updates):
    data = {
        "model": "unit-model",
        "messages": [{"role": "user", "content": "answer"}],
    }
    data.update(updates)
    return ChatCompletionRequest.model_validate(data)


def _config(tool="qwen3_coder", reasoning=None):
    return SimpleNamespace(tool_call_parser=tool, reasoning_parser=reasoning)


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def test_json_object_response_format_reaches_sampling_params():
    spec = chat_request_to_genspec(
        _request(response_format={"type": "json_object"}), {}, _config()
    )

    assert spec.sampling_params.guided_decoding == {
        "kind": "json_schema",
        "schema": {"type": "object"},
        "strict": False,
    }


def test_json_schema_response_format_preserves_schema_and_strictness():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    spec = chat_request_to_genspec(
        _request(response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": schema},
        }),
        {},
        _config(),
    )

    assert spec.sampling_params.guided_decoding == {
        "kind": "json_schema", "schema": schema, "strict": True,
    }


def test_qwen3_json_constraint_starts_after_reasoning():
    spec = chat_request_to_genspec(
        _request(response_format={"type": "json_object"}),
        {},
        _config(reasoning="qwen3"),
    )

    assert spec.sampling_params.guided_decoding["start_after"] == "</think>"


def test_required_tools_build_qwen_structural_constraint():
    spec = chat_request_to_genspec(
        _request(tools=_tools(), tool_choice="required"),
        {},
        _config(reasoning="qwen3"),
    )
    constraint = spec.sampling_params.guided_decoding

    assert constraint["kind"] == "tool"
    assert constraint["style"] == "qwen_3_coder"
    assert constraint["tool_choice"] == "required"
    assert constraint["reasoning"] is True
    assert constraint["force_reasoning"] is True
    assert constraint["tools"][0]["function"]["strict"] is True


def test_required_tools_need_a_declared_tool():
    with pytest.raises(ValueError, match="at least one tool"):
        chat_request_to_genspec(
            _request(tools=[], tool_choice="required"), {}, _config()
        )


def test_response_format_rejects_malformed_json_schema_envelope():
    with pytest.raises(ValueError, match=r"json_schema\.schema"):
        chat_request_to_genspec(
            _request(response_format={
                "type": "json_schema", "json_schema": {"name": "missing-schema"},
            }),
            {},
            _config(),
        )


def test_required_tools_reject_unsupported_posthoc_parser():
    with pytest.raises(ValueError, match="no XGrammar structural grammar"):
        chat_request_to_genspec(
            _request(tools=_tools(), tool_choice="required"),
            {},
            _config(tool="gemma4"),
        )


def test_constrained_request_rejects_stop_strings_that_can_truncate_json():
    with pytest.raises(ValueError, match="stop is not supported"):
        chat_request_to_genspec(
            _request(response_format={"type": "json_object"}, stop="}"),
            {},
            _config(),
        )
