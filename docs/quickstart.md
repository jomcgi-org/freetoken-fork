# Quick start

Assumes FreeToken is installed — see [install.md](install.md).

## Launch a server

```bash
ft serve --model ~/models/Qwen3.6-35B-A3B
```

`--model` also takes a Hugging Face repo id. Everything else — dtype, attention
and MoE backends, cache sizes, tool-call and reasoning parsers — resolves from
the checkpoint and the GPU; see [cli.md](cli.md) for the flags. The server is
ready when the log reaches `API server is ready to serve on 127.0.0.1:1919`.

## Send a request

Check what is being served:

```bash
curl http://127.0.0.1:1919/v1/models
```

Then use that id as the `model` field:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "What is a Mixture-of-Experts model?"}],
    "max_tokens": 256,
    "stream": true
  }'
```

FreeToken serves the OpenAI API (`/v1/chat/completions`, `/v1/responses`,
`/v1/models`) and the Anthropic API (`/v1/messages`,
`/v1/messages/count_tokens`), so a client library for either works by pointing
its base URL at the server. 

### Request priority

OpenAI-compatible requests may include an integer `priority` field, or send the
`x-request-priority` HTTP header. Higher values are admitted first and the header
wins when both are present. Requests default to priority 0.

Waiting time adds one effective priority point every 30 seconds. Configure the
interval with `--priority-aging-seconds`; set it to 0 to disable aging. Priority
only orders waiting prompt admission. It does not preempt an in-flight forward or
a request already running in decode. A chunked prompt returns to the waiting queue
between chunks, so a higher-priority request can win at that boundary. Running
request preemption would require the future prefill-decode mixing seam.

## Chat in the terminal

A simple TUI to interact with the server:

```bash
ft shell                                    # attach to the server above
ft shell --model ~/models/Qwen3.6-35B-A3B   # start an engine and chat, one process
```

`/help` lists the in-shell commands. Attach mode needs no GPU, so it also drives
a server on another machine (`--server URL`).

## Use a coding agent

```bash
ft launch claude   # claude / codex / dsh / hermes / openclaw / opencode
```

Writes that agent's provider config, installs its CLI if missing, and starts it
against your server. `--dry-run` previews the changes.
