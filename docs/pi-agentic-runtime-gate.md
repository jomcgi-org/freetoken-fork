# Original and optimized runtime comparison with Pi

The completed 2026-09-06 comparison takes **23.1% less measured task wall time**
with the optimized stack: 1506.667 seconds (25m07s) versus 1158.016 seconds
(19m18s) across four measured tasks per runtime. All eight measured tasks and
four warmups pass all three cumulative requirement checks. The original service
was restored and verified with a real completion.

| Start order | Baseline task wall | Optimized task wall | Reduction |
| --- | ---: | ---: | ---: |
| A/B, two tasks each | 665.618 s | 633.515 s | 4.8% |
| B/A, two tasks each | 841.049 s | 524.501 s | 37.6% |
| Total, four tasks each | 1506.667 s | 1158.016 s | 23.1% |

This is observed agentic task throughput, with substantial model-behavior
variation. The optimized tasks use 67 model calls and 22752 output tokens,
versus 85 calls and 26999 tokens for the baseline. The final baseline task
spends five failed test runs revising an incorrect expectation before removing
that test, finishing in 514.508 seconds with 29 calls. Its full elapsed time
remains in the comparison. Both start orders improve, but their spread and
unequal generation prevent treating 23.1% as an isolated engine speedup.
The first order is only 4.8% faster despite similar output-token totals.

Measured tasks contain nine failed tool commands in the baseline and five in
the optimized runtime. No independent grader repair prompt was required.
Final implementations, final test results, and repair sequences were reviewed;
some repairs correct implementation errors, others correct invalid model-written
tests. All 229 HTTP completions match readiness plus the recorded Pi calls;
there are no inference errors. This one Cache fixture is a limited quality
check, not proof of broad quality equivalence.

The [complete record](../bench/results/4090-pi-runtime-wall-20260906.json)
contains arithmetic, all twelve transcripts and event records, final workspaces,
full journals, source identities, the frozen controller and client, and recovery
evidence. Each of its 73 raw files retains exact text and a SHA-256 hash.
The earlier partial arm reviews are retained as evidence captured during the
run; the complete record and this document supersede their pending status.

This gate measures the existing throughput stack on a continuing coding task.
It does not enable the native shared-route decode experiment, change routing,
or introduce quantization. The baseline is the frozen original `3a67403`
runtime and native binary. The optimized runtime is frozen at `c0775ea`, using
the already-qualified CPU input reuse binary, buffered staged GPU DISK prefill,
and mmap HOT staging. All source trees remain clean and unchanged.

The [controller](../bench/pi-agentic-runtime-driver.py) runs baseline, optimized,
optimized, baseline. Each new server performs an identical readiness completion,
then one warmup and two measured Pi sessions. Every session starts the same
fixture and retains conversation history through all three requirements. Failed
tasks and warmups remain in the record, and a failure does not silently trigger
a replacement attempt. There are four measured sessions per runtime.

Both servers use capacity one, graph size one, a 65536-token KV reservation,
FP8 K/V as resolved by the existing backend, and radix prefix reuse. The KV
ladder, disk prefix cache and persisted HOT plans are disabled. The same
automatic HOT controller remains active. Host page cache is retained across
starts. Actual placement, expert slots, KV allocation, CPU thread count,
precision, and graph geometry must match before an arm can run. These fixed
budgets differ from the production ladder policy; the comparison measures the
runtime stack under this controlled agentic configuration.

Pi runs on the Mac using the pinned client from #38. The model is reached
through a dedicated SSH tunnel to port 18090 on node-4. Existing port 8090
tunnels are left intact. The output directory is archived after each arm and
the same path is reused for the next one, preserving the working-directory
text in Pi's system prompt. Output limits, prompts, tools, sampling and grading
are identical. Each task allows 900 seconds, 30 model calls, 8192 output tokens
per call, and one independent-check repair per stage.

`--moe-collect-stats`, GPU step timing and client streaming traces are off.
The existing inexpensive cache usage report stays enabled. Full worker I/O
snapshots and journals are collected outside client timing. Those counters
cover warmup and measured sessions together; they do not isolate expert I/O or
attribute time to prefill versus decode.

The original runtime still calls its disk/PLE statistics helpers from status
logging even with the diagnostic flag off. The optimized runtime guards those
calls, as part of the existing telemetry changes. Accordingly, "off" describes
the requested flags in both arms, not the absence of every legacy diagnostic
operation in the baseline. This combined comparison cannot isolate the speedup
from removing that work.

A remote systemd lease expires if the Mac disconnects or stops sending its
20-second heartbeat for 90 seconds. Its `ExecStopPost` stops the benchmark
server and verifies a real completion from the original service. A four-hour
lease bound and a per-server runtime limit provide additional termination
bounds. The controller records restoration even after a failed arm. It stops
its own tunnel and Pi process tree on exit. Infrastructure failures stop the
gate; retained model/check failures continue through the declared opposite arm.

Preflight verifies identities and inactivity of other benchmarks without
stopping the original server:

```sh
python3 bench/pi-agentic-runtime-driver.py \
  --run-id astra-pi-agentic-runtime-20260906 \
  --output-dir /private/tmp/astra-pi-agentic-runtime-20260906 \
  --pi /private/tmp/astra-pi-cli/node_modules/.bin/pi \
  --preflight
```

Commit the driver before preflight, then launch the same command without
`--preflight` after its checks pass. A run refuses to overwrite existing arm
records. Review both start orders, task success, all attempted wall time,
individual stages, model calls, tool failures, and cache usage before claiming
an improvement. Four successes per arm are still a small coding evaluation;
they do not prove broad quality noninferiority.

TurboQuant remains a separate possible experiment. The deployed Qwen Flash
has 12 full-attention layers and 36 linear-attention layers; conventional K/V
compression does not compress the experts or linear state. The controlled
64K pool uses about 0.80 GiB for K/V, so ideal 4-bit storage would save roughly
0.4 GiB before overheads. The [TurboQuant paper](https://arxiv.org/abs/2504.19874)
reports quality neutrality on its tests, but a [later vLLM evaluation](https://vllm-project.github.io/2026/05/11/turboquant.html)
found reasoning/coding degradation with aggressive variants and lower
throughput than FP8 on its H100 workloads. Those are not this model's results.
Any implementation needs its own quality and non-debug wall-time gate; this
comparison retains FP8 unchanged.

Nine focused driver checks pass on Linux and macOS. Live systemd probes also
verify both normal EOF recovery and the 90-second missing-heartbeat timeout.
Both probes confirm a real completion from the original service and preserve
its invocation, since it was already serving. The timeout probe exits
unsuccessfully as expected and still runs recovery. The
[complete recovery record](../bench/results/4090-pi-runtime-recovery-20260906.json)
retains both commands, exit status, output, and verified original completion.

The gate ran from 03:11:30 UTC through verified restoration after 04:32 UTC.
All four starts match 3753 expert slots, 1024 KV pages, 2296 protected HOT rows,
14 CPU threads, capacity one and graph size one. The 64K K/V allocation is
0.80 GiB. This differs from the earlier naive-cache experiments.

Summarize the archived records after collection:

```sh
python3 bench/pi-agentic-runtime-summary.py \
  /private/tmp/astra-pi-agentic-runtime-20260906 >pi-runtime-summary.json
```

To reconstruct those inputs from the published record in a fresh local directory:

```python
import hashlib
import json
from pathlib import Path

record = json.loads(Path("bench/results/4090-pi-runtime-wall-20260906.json").read_text())
root = Path("pi-runtime-reconstructed")
root.mkdir(exist_ok=False)
for name, item in record["raw_files"].items():
    data = item["text"].encode()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
```

Then pass `pi-runtime-reconstructed` to the summarizer. It rejects incomplete
schedules, mismatched configuration or identities, enabled diagnostic flags and
inconsistent successful checks. It retains all twelve sessions, excludes only
the declared warmups from comparison, and omits speedup if measured tasks fail.
Worker storage counters cover warmup and measured tasks together. The analysis
cannot establish broad quality equivalence or attribute time to individual
runtime changes.

Fifteen focused summarizer checks pass on both Linux and macOS, including
rejection of incomplete or inconsistent evidence and retention of failed-task
time. The [Linux validation record](../bench/results/4090-pi-runtime-summary-validation-20260906.json)
confirms these checks ran after model timing and verified restoration, with the
same original serving invocation before and after. Nine focused controller
checks and the live recovery probes also passed before the run. This analysis
changed neither the frozen controller nor client used for collection.
