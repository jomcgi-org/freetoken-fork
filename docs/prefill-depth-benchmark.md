# Real-source prefill depth comparison

`bench/prefill-depth.py` prepares one fixed manifest and measures cold prefill,
an immediate cached repeat, and a separate decode request after each document.
It addresses the measurement portion of fork issue #80. It does not change the
serving defaults or qualify a new profile by itself.

Prepare once on the serving machine, before timing any arm:

```sh
python bench/prefill-depth.py prepare \
  --source . --tokenizer /path/to/flash-e2m1.ftw \
  --depths 8000 32000 100000 --runs 3 --output manifest.json
```

The manifest contains excerpts from the selected source tree and unique prefixes
per depth and repetition. Its rendered inputs are within 256 tokens below the
requested depths. All arms must use the same manifest. The model must support
the largest input plus the 192-token output allowance; the node-4 comparison
keeps its existing 100352-token KV reservation.

Start an arm with the qualified launch script and override only the chunk size
being tested. For the cold sweep, disable disk prefix persistence in **every**
arm with `--kv-disk-cache-gib 0`, retain the ordinary radix cache, and restart
between arms. This means cold **prefix** state, not flushed operating-system
file caches. Wait for `/health` JSON `status` to become `ok`; HTTP 200 alone also
occurs during model loading.

```sh
python bench/prefill-depth.py measure \
  --manifest manifest.json --arm chunk-2048 \
  --base-url http://127.0.0.1:18090 --output chunk-2048.jsonl
```

Run 2048, 4096 and 8192, then repeat the control or reverse the order. Record the
exact source revision, native extension, model, launch command, startup memory
budget and layer residency beside each result. The native CPU task descriptor
accepts up to 2147483647 tokens, so it does not bind these candidate chunk sizes;
actual host and device allocations do.

Each result retains the complete streamed response, token usage, finish reason,
request hash, first-generated-text latency, total wall time and answer check.
The decode-rate estimate excludes latency before the first generated text and
uses completion tokens minus one divided by the interval to the last text
chunk. It is a client-observed estimate, not an engine kernel timer. Role and
keepalive frames do not end the first-token wait. The repeat measures immediate
in-memory reuse; it does not establish disk-restore or tool-continuation speed.

Reject truncated, failed or incorrectly copied answers. Check `cold_valid` and
actual cached-token counts before accepting a cold measurement. Compare request
hashes and output counts across arms, retaining differences rather than silently
discarding slow or failed runs. JSON copying is a narrow fidelity check, not a
general quality gate. Chunk changes can change floating-point rounding and
generated text. A finalist still needs the existing matched multi-turn/coding
checks, a disk-prefix restore check, and repeated measurements at long context.

Do not change the default until the harness-root persistence gap in #81 is
addressed or its effect on the chosen profile has been explicitly qualified.
Larger chunks can cause more system roots to fall inside a final chunk, where
the existing implementation does not persist them independently.
