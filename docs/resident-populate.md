# Skip redundant CPU populate reads

The completed 4090 model comparison regresses: twenty-four measured responses
take 518.987 seconds with the flag off and 598.375 seconds with it on, **15.3%
more client wall time**. Both start orders regress. Keep the option disabled.
The component improvement below does not translate into inference throughput.

The CPU prefill path warms selected file-backed experts with buffered reads into
a reusable scratch buffer. It discards the read data and computes from the original
bank mappings. A warm request can therefore copy gigabytes of already resident
weights solely to prepare those same mappings for computation.

`FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT=1` enables an experimental shortcut in
that preparation step. Before each scratch-sized read, it checks the pages covering
the exact requested mapping range. It skips the read only when every page is
reported resident. A partly resident range, an unavailable probe, or a file owned
by another user retains the existing buffered read. The default remains disabled.
Neither target routing nor packed weights, scales, native arithmetic, or expert
placement changes. The existing scratch buffer and memory geometry remain intact.

Linux [mincore](https://github.com/torvalds/linux/blob/v6.8/mm/mincore.c#L147)
may conceal residency for mappings the caller does not own or cannot write. The
experiment conservatively probes only owned files. A successful result can also
be stale immediately. Native computation retains its original file-backed pointers
and can demand-fault if a page leaves RAM after the check. Probe failure chooses
the existing read path. The hint never selects different model data.

Each populate call owns a bounded bitmap, including calls from the background
prefill thread. A probe covers at most 32 MiB before reusing that bitmap; larger
requests are checked in pieces. Address calculation includes the bank mapping
offset and an unaligned tensor view. The returned populate-byte counter counts
actual scratch reads, while no extra diagnostic counters or device readbacks are
introduced. Probe time is part of any measured client wall time.

This is separate from the negative cache-aware GPU staging experiment. GPU staging
must still copy every selected weight into VRAM. CPU populate data is discarded,
so this experiment can eliminate a read without providing replacement bytes or
using direct I/O. Changed cache-access patterns and stale hints can still reduce
performance; no improvement is assumed from fewer copied bytes.

All eleven focused checks pass on Linux with no skips: seven probe checks and
four real file-bank checks. They cover unaligned boundaries, the defined
residency bit, bounded queries, failed probes after a resident answer, foreign
ownership, unsupported platforms, the default read path, real warm-file skipping,
mixed hints, fallback reads, and unchanged mapped bytes. The seven hermetic
probe checks also pass on macOS. The [Linux validation record](../bench/results/4090-populate-resident-validation-20260906.json)
retains exact sources, command, output and unchanged serving state. Validation
ran after the Pi comparison finished and the original service was restored.
These checks establish the probe and mapped-byte behavior. Complete model timing
is reported below.

The completed Pi runtime comparison used frozen sources without this option.
The model comparison uses the resident-skip flag off/on on the same prepared
sources, mapping geometry and native binary. It retains complete requests in
both orders and all output checks. Its negative result supports leaving the
option disabled.

The [component benchmark](../bench/resident-populate.py) ran on Linux after
the correctness checks. It creates a private 256 MiB file, selects alternating
2 MiB rows, and compares both flag orders under warm, cold, and mixed preparation.
Cache advice targets only that file. It verifies the count of fully resident
selected rows before each sample, times population plus SHA-256 consumption from
the original mapping, and checks the resulting bytes. The checksum is a memory
consumer for this component test, not an MoE compute benchmark. Probe time and
subsequent faults are included; file preparation is outside timing. Source hashes,
raw samples, scratch bytes copied, and process I/O deltas are retained.

Run from a clean committed worktree with its Python package selected, after other
timed work has finished. Write output outside the worktree:

```sh
PYTHONPATH=python python bench/resident-populate.py \
  --directory /var/lib/longhorn/nvme-02/freetoken/tmp \
  > /var/lib/longhorn/nvme-02/freetoken/results/resident-populate-component.json
```

The [completed Linux component record](../bench/results/4090-populate-resident-component-20260906.json)
uses 30 samples, including six declared warmups. All samples consume the same
selected bytes and match the expected checksum. Four measured off/on pairs per
cache condition alternate their start order:

| Selected-row residency | Flag off, mean | Flag on, mean | Total wall reduction |
| --- | ---: | ---: | ---: |
| Fully warm | 64.17 ms | 54.57 ms | 15.0% |
| Cold | 155.90 ms | 157.44 ms | -1.0% |
| Half warm | 106.32 ms | 103.37 ms | 2.8% |

All four warm and mixed pairs improve. Three of four cold pairs regress slightly.
Warm population copies no scratch bytes with the flag on; mixed population copies
half as many. Physical storage reads match between modes in each condition.
These timings include the subsequent checksum consumer and probe overhead, but
exclude file preparation. They establish a component effect, not an inference
speedup under model memory pressure.

The component process succeeded with serving stopped and no GPU workers. Its
reused recovery helper rejected the experiment's run-name prefix, so automatic
restoration failed. The original service was explicitly started and a separate
detached helper with a valid name verified health and an OK completion. The full
record retains the failure and remediation. Future gates must validate the exact
recovery command before stopping serving.

The whole-model comparison includes CPU prefill below the 1024-token staged-GPU
threshold alongside longer requests, with complete response checks and both
start orders.

The sustained client accepts `--mixed-prefill` for this model gate. It
balances short and long prompts across JSON, prose, source excerpt and measured
position. Short backgrounds use 128 tokens; long backgrounds retain the existing
1400-token excerpt. Output requirements and caps are unchanged. The client uses
actual server prompt usage to require 2-1023 tokens for the short band and at
least 1024 for the long band. A wrong band retains the full response and time,
but marks that request ineligible through its completion field. Prefix caching
must stay disabled for this gate so the full prompt length describes prefill
work. Twenty-four focused protocol checks pass on Linux and macOS. The
[complete validation record](../bench/results/4090-populate-mixed-client-validation-20260906.json)
retains the first incorrect offline token-count check and its correction to the
server render-then-encode path. Verified prompts contain 424-575 tokens in the
short band and 1700-1844 in the long band. No model run has used this mode yet.

The [model driver](../bench/resident-populate-wall-driver.py) executed
four starts with the resident flag off/on/on/off. Each uses the same frozen
`e865d19` runtime and existing input-reuse native binary. The driver uses the
frozen recovery helper installed by #39, verifies its hash, and records both
sources in the resulting artifact. Capacity one, graph one, 64K FP8 K/V,
buffered staged GPU prefill, mmap HOT staging, and automatic HOT adaptation
remain fixed. Prefix caches and diagnostic flags are disabled. Each start runs
four warmups, twelve measured complete responses and eight fidelity questions.
Actual prompt lengths, source identities, native mappings, execution geometry
and HTTP request counts must match before a result can qualify.

Before launching, use `--preflight` and execute the exact recorded recovery
command through a systemd `ExecStopPost` while original serving remains active.
Retain the successful completion as `recovery-probe.json` in the run directory.
The driver refuses a prior model run and requires the probe's serving invocation
to match current serving. The enclosing driver unit must use that same recovery
command, a four-hour runtime bound and a 600-second stop timeout. This preparation
is not yet a model performance result.

The [first model attempt](../bench/results/4090-populate-resident-wall-startup-failure-20260906.json)
failed before readiness or any timed response because the dedicated worktree
lacked `_ple_uring`. Its pretested recovery command restored original serving
and verified a real completion. This infrastructure failure provides no model
performance result. The attempt and its journals are retained.

The retry uses a new `-v2` run directory. Preflight now verifies and imports the
unchanged PLE, pinned-tensor and UFFD support binaries as well as CPU MoE,
without initializing CUDA. The [revised preflight record](../bench/results/4090-populate-resident-runtime-preflight-v2-20260906.json)
confirms those imports and hashes pass on Linux. The support sources match the
working runtime.
Startup records retain the server invocation before readiness, and the worker
must map the expected native PLE reader before timing. Failure journals are
retained even when readiness fails. These checks address the missing-dependency
failure; the retry completed all four starts.

The revised comparison launched at 05:15:45 UTC on 2026-09-06 and reached
inference. Its [partial startup record](../bench/results/4090-populate-resident-wall-startup-v2-20260906.json)
confirms a successful recovery probe, expected native reader mappings, and
valid short/long prompt bands in the completed responses at capture. The first
start has 4045 expert slots, 1024 KV pages and 2296 protected HOT rows; every
later start must match. This naive-cache geometry differs from the Pi gate's
3753 slots. The systemd driver ran independently of the local CLI. The complete
record below supersedes this partial capture.

## Completed model comparison

The [complete record](../bench/results/4090-populate-resident-wall-20260906.json)
retains all four off/on/on/off starts, their manifests, all sixty-four response
rows including sixteen warmups, thirty-two fidelity answers, journals, source
hashes, analysis code and manual prose review. The final collection reconstructs
every raw file, verifies its hash, reruns the analysis and independently checks
the primary wall-time totals.

| Measured client workload | Flag off | Flag on | Wall-time increase |
| --- | ---: | ---: | ---: |
| Complete JSON, mean | 24.509 s | 26.232 s | 7.0% |
| Complete prose, mean | 18.740 s | 23.633 s | 26.1% |
| Short-prompt responses, mean | 19.397 s | 20.733 s | 6.9% |
| Long-prompt responses, mean | 23.852 s | 29.131 s | 22.1% |
| Fixed 24-request mix, total | 518.987 s | 598.375 s | 15.3% |

The first matched order regresses 23.8%; the reversed order regresses 7.9%.
Sixteen of twenty-four paired responses take longer. Source-balanced first and
second halves regress 17.6% and 12.2%. Twelve identical-text pairs, including
eleven JSON pairs, also regress 11.8%, so generated prose variation does not
explain the entire effect. The on setting generates fewer measured tokens
overall, 8272 versus 8378, while taking more time.

For short prompts, mean client-observed first-text time improves from 5.813 to
5.225 seconds, but the rest of the response increases from 13.584 to 15.508
seconds. Across the full mix those intervals are 6.591 to 6.692 seconds and
15.033 to 18.241 seconds. These are client boundaries, not exact GPU phase timings.

Measured whole-worker logical read bytes fall from 514.00 to 341.07 GiB while
physical reads rise from 64.18 to 92.74 GiB, a 44.5% increase. Physical reads
increase in the first order and decrease in the reversed order, while both
orders take longer. This suggests changed cache behavior, but the counters do not identify the eviction
mechanism or isolate expert, PLE, prefill and decode traffic. The flag changes
only CPU population, yet subsequent long requests can inherit changed cache
conditions. Fewer scratch reads are not sufficient evidence of a serving gain.

All JSON outputs pass strict value, type, key-order and multiplicity checks.
All prose finishes in three paragraphs with two or three sentences each. Manual
review retains omissions and unsupported explanatory connections in both modes;
one on-setting response omits the valid HOT-owner requirement and others
overstate output-distribution guarantees. All four starts return identical
fidelity answers and the same 7/8 score, failing code trace with 108 rather than
68. These limited observations do not establish broad quality equivalence.

All one hundred expected HTTP requests complete successfully. The second
start logs a detokenizer-exit error during intentional shutdown, after its
twenty-five successful requests; its surrounding journal is retained. There
are no inference errors. Original serving is restored at invocation
`c802252497d345779fc34390544dfe46`, with health and a real OK completion verified.
The original worker maps the expected native binary. The driver invocation
`ec727c12459748258c11289a2290296d` has four completed-arm records and a matching
systemd successful-deactivation event. This journal evidence survives transient
unit collection, unlike the empty/default properties of an unloaded unit.
