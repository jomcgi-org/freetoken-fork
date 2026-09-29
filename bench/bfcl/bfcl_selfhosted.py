"""Drop-in replacement for the `bfcl` CLI that registers a self-hosted
OpenAI-compatible chat-completions endpoint as an FC model.

Env:
  OPENAI_BASE_URL      e.g. http://host:port/v1   (read by OpenAICompletionsHandler)
  OPENAI_API_KEY       any string (SDK requires one)
  BFCL_MODEL_NAME      model name sent in the request body (served model name)
  BFCL_REGISTRY_NAME   name used for --model and result dirs (default: local-qwen-FC)
  BFCL_MAX_TOKENS      optional max_tokens per call (BFCL sends none by default)
  BFCL_EXTRA_BODY      optional JSON merged into request body, e.g.
                       '{"chat_template_kwargs":{"enable_thinking":false}}'

Usage: python bfcl_custom.py generate --model local-qwen-FC ...   (same as `bfcl`)
"""
import json
import os

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


class SelfHostedFCHandler(OpenAICompletionsHandler):
    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        kwargs = {"messages": message, "model": self.model_name, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        if os.getenv("BFCL_MAX_TOKENS"):
            kwargs["max_tokens"] = int(os.environ["BFCL_MAX_TOKENS"])
        if os.getenv("BFCL_EXTRA_BODY"):
            kwargs["extra_body"] = json.loads(os.environ["BFCL_EXTRA_BODY"])
        return self.generate_with_backoff(**kwargs)

    def _parse_query_response_FC(self, api_response):
        response_data = super()._parse_query_response_FC(api_response)
        # Replay a clean assistant message (no reasoning / null SDK fields) and log
        # reasoning to the result file, like the DeepSeek/Grok handlers do.
        msg = api_response.choices[0].message
        hist = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            hist["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        response_data["model_responses_message_for_chat_history"] = hist
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if reasoning:
            response_data["reasoning_content"] = reasoning
        return response_data


REGISTRY = os.getenv("BFCL_REGISTRY_NAME", "local-qwen-FC")
MODEL_CONFIG_MAPPING[REGISTRY] = ModelConfig(
    model_name=os.getenv("BFCL_MODEL_NAME", "qwen"),
    display_name=f"{REGISTRY} (FC)",
    url="http://localhost",
    org="self-hosted",
    license="n/a",
    model_handler=SelfHostedFCHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,  # OPENAI_COMPLETIONS style rewrites '.' -> '_' in tool names
)

if __name__ == "__main__":
    from bfcl_eval.__main__ import cli

    cli()
