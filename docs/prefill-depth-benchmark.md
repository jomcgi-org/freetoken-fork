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

## Initial node-4 screening, 2026-09-22

Runtime `4fbc4ebf3f7d0c4039893b0471faf231c7a15ba0`, RTX 4090, 61.91 GiB host
RAM, 100352 reserved KV tokens, 14 CPU threads and 6 GiB protected HOT budget.
All arms kept 20 PINNED and 28 DISK layers, 82 protected rows per DISK layer,
3589 expert slots, and 2.12 GiB free GPU memory after graph capture. The CPU
extension SHA-256 was
`c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f`.

The order was 2048, 8192, 4096, 2048. Each server started from its static expert
profile with empty prefix state. Each arm ran an approximately 8k source prompt,
its cached repeat and a separate decode request, then the same sequence at 32k.
Actual input counts were 7959 and 31958. This is one fixture per depth and two
control observations, not a confidence interval or a selected serving default.

| Chunk size / order | Cold TTFT, 8k | Cold TTFT, 32k | Decode during cold 32k answer | Independent decode after 32k |
| --- | ---: | ---: | ---: | ---: |
| 2048 / first | 46.61 s | 82.38 s | 11.87 tok/s | 29.16 tok/s |
| 8192 / second | 12.86 s | 52.30 s | 5.30 tok/s | 29.52 tok/s |
| 4096 / third | 20.62 s | 70.26 s | 10.12 tok/s | 31.09 tok/s |
| 2048 / last | 33.61 s | 96.37 s | 10.21 tok/s | 29.12 tok/s |

All 24 responses passed the JSON-copy checks. The cold and repeat answers each
used 81 output tokens; independent decode used 451. All cold requests reported
zero cached tokens. The 8192 cold 32k request completed in 67.91 s versus
89.22 and 104.32 s for the controls, but its immediate decode and cached repeat
were slower: repeat wall was 4.70 s versus 3.63 and 3.68 s. Independent subsequent
decode recovered to the control rate. That transient cost remains part of the
qualification, not a discarded measurement.

The governor charged prefill scratch of 0.37, 0.64 and 1.18 GiB for chunks 2048,
4096 and 8192 respectively. Expert layer placement was unchanged and no host
pressure warning appeared in the saved arm journals. These startup estimates
do not substitute for the pending 100k pressure measurement.

Private full-response artifacts, manifests, launch commands and journals are in
`node-4:/var/lib/longhorn/nvme-02/freetoken/results/prefill-depth-20260922/`,
using `screen3-*.jsonl`. Earlier controller setup failures are preserved under
different names and excluded from this table. The original serving service was
restored successfully after the sweep.

## 100k screening and host pressure

The same runtime then ran one fixed 99,959-token prompt with 81 output tokens,
its immediate repeat, and a separate 451-token decode request. The order was
8192, 8192 with `--moe-hot-adapt-prefill-run-cap-frac 0.1`, then 2048. The cap
limits expert-cache swaps during a prefill run to a fraction of the HOT budget;
zero disables this additional cap. All other profile settings stayed fixed,
including disabled disk-prefix persistence and enabled in-memory radix reuse.

| Chunk / prefill swap cap | Cold TTFT | Cold wall | Cold answer decode | Repeat wall | Independent decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 / disabled | 167.77 s | 175.43 s | 10.71 tok/s | 8.36 s | 25.13 tok/s |
| 8192 / 0.1 | 141.73 s | 147.38 s | 14.65 tok/s | 8.70 s | 24.03 tok/s |
| 2048 / disabled | 267.03 s | 272.32 s | 15.77 tok/s | 6.30 s | 26.09 tok/s |

All nine responses passed, with identical request hashes, answer bytes and output
counts across arms for each phase. Cold requests had zero cached tokens; repeats
had 99,904. Repeat TTFT was 2.67, 2.68 and 2.72 seconds respectively, so the
repeat regression was after first text. Repeat decode estimates were 14.38,
13.54 and 23.14 tok/s. The capped candidate improved cold TTFT by 1.88x, but its
38% longer repeat wall and lower subsequent decode rate prevent a claim that it
preserves decode performance. These are single observations, not a qualified
default or a confidence interval.

Two-second samples of `/proc/meminfo`, `/proc/pressure`, `/proc/vmstat` and
`/proc/diskstats` were correlated with each cold request's client interval:

| Chunk / cap | Sampled / request seconds | Min available RAM | Memory full stall | I/O full stall | Main NVMe reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 / disabled | 164.4 / 175.4 | 28.65 GiB | 14.17% | 26.69% | 76.86 GiB |
| 8192 / 0.1 | 144.2 / 147.4 | 28.47 GiB | 9.43% | 21.05% | 59.44 GiB |
| 2048 / disabled | 270.5 / 272.3 | 29.47 GiB | 2.20% | 10.85% | 51.25 GiB |

Stall percentages use deltas of the kernel's cumulative `full` pressure counters,
not averages of its rolling averages. NVMe reads use sector-counter deltas for
`nvme1n1`. These are whole-host counters, not process-exclusive attribution. The
monitor started after the first request began, and two-second sampling also
omits interval edges. Available memory alone did not capture the pressure:
larger chunks were faster despite more sampled disk traffic and stall time.
This motivates the host-budget diagnosis in #82, but does not prove a specific
memory-governor fix.

Artifacts use `long1-*-chunk-*.jsonl`, corresponding command/journal files,
`long1-host-pressure.jsonl` and `summarize-long-pressure.py` in the same private
results directory. A matched three-turn continuation comparison and additional
prefill-to-decode transition measurements remain pending. The harness-root
restart check also exposed a separate tokenizer-template rejection of system-only
messages; PR #85 addresses that alongside final-chunk snapshots and remains
pending real serving validation. No serving default has been changed.
