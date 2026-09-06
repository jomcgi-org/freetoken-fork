# Original and optimized runtime comparison with Pi

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

The gate launched on 2026-09-06 at 03:11:30 UTC. The
[first-arm startup record](../bench/results/4090-pi-runtime-launch-20260906.json)
confirms the baseline native mapping, both diagnostic flags off, one-request
capacity, FP8 K/V, and successful readiness completion. Prefix caching leaves
3753 expert slots with 1024 KV pages; HOT residency remains 2296 rows. This
differs from the earlier naive-cache experiments. Later arms must match this
geometry before comparison. The record is partial, with no measured speedup
yet; the controller and recovery lease were confirmed live at capture.

After all four arms finish and serving is restored, summarize the archived records:

```sh
python3 bench/pi-agentic-runtime-summary.py \
  /private/tmp/astra-pi-agentic-runtime-20260906 >pi-runtime-summary.json
```

The summarizer rejects incomplete schedules, mismatched client configuration,
server identities or geometry, enabled diagnostics, and inconsistent successful
checks. It retains all twelve sessions, reports the eight measured sessions and
both start orders, and omits speedup and checked-task throughput when measured
tasks fail. Worker storage reads include warmup and are labelled accordingly.
Input hashes make the summary traceable to its raw records. Review the full
journals, model outputs, and final files separately; arithmetic and fixture checks
cannot establish broad quality equivalence. This analysis runs after collection
and changes neither the frozen client nor the runtime controller.

Fifteen focused summarizer tests pass on macOS. Its CLI also refuses to summarize
the live incomplete gate. Linux validation remains pending until the timed model
comparison ends, to avoid competing with the measured worker.

The [first baseline arm review](../bench/results/4090-pi-runtime-r1-review-20260906.json)
records all three successful sessions, their independent checks, final test
results, source hashes and repair behavior. The two measured tasks total 665.618
seconds. Its archived journal accounts for exactly 60 requests, matching readiness
plus 59 Pi model calls, with no inference errors. This is partial evidence; it
contains no optimized comparison or claim of broad quality equivalence.

The [first optimized arm review](../bench/results/4090-pi-runtime-r2-review-20260906.json)
also records three successful sessions and exactly the expected 52 requests,
without inference errors. Its two measured tasks total 633.515 seconds, versus
665.618 for the first baseline arm, a provisional 4.8% reduction. The reverse
start order remains pending. The optimized log omits the legacy disk-statistics
fields. Review distinguishes implementation errors from mistaken model-written
test expectations; both kinds of repair remain in the elapsed task time.

The [second optimized arm review](../bench/results/4090-pi-runtime-r3-review-20260906.json)
retains another three successful tasks and 52 expected requests without inference
errors. Its measured tasks total 524.501 seconds, with fewer generated tokens and
model calls than the first optimized arm. That shorter unpaired time is not a
new runtime gain. The final baseline start matches the same geometry; its results
and the complete paired analysis remain pending.
