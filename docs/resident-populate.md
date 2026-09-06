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
Next measure the component, then compare the resident-skip flag off/on on the
same prepared sources, mapping geometry and native binary. Use complete requests
in both start orders, retain all failures, and include both cache warmth and
worker storage traffic. Keep this option disabled until complete wall-time and
output checks justify using it.

The [component benchmark](../bench/resident-populate.py) is prepared for Linux timing
after the correctness checks. It creates a private 256 MiB file, selects alternating
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

Only syntax and CLI checks have run locally for this benchmark. Linux execution
remains pending; neither this preparation nor checksum parity establishes a model
throughput improvement.
