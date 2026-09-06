# Skip redundant CPU populate reads

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
No native or model wall-time result is available for this experiment.

The completed Pi runtime comparison used frozen sources without this option.
The component result below justifies comparing the resident-skip flag off/on
on the same prepared sources, mapping geometry and native binary. Use complete requests
in both start orders, retain all failures, and include both cache warmth and
worker storage traffic. Keep this option disabled until complete wall-time and
output checks justify using it.

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

The whole-model comparison remains pending. It must include CPU prefill below
the 1024-token staged-GPU threshold, alongside longer requests, and retain
complete response checks and both start orders. Keep the option disabled until
that comparison establishes its effect on client wall time.

The sustained client now accepts `--mixed-prefill` for the next model gate. It
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

The [model driver](../bench/resident-populate-wall-driver.py) is prepared for
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
failure; the model wall-time comparison remains pending.
