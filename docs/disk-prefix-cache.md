# Disk-backed hybrid prefix cache

Enable the lane with both flags:

```text
--kv-disk-cache-dir /nvme/freetoken-prefixes --kv-disk-cache-gib 1024
```

Harness signatures are configurable as repeated `kind=prefix` entries. Supplying
the flag at least once replaces the built-in OpenCode and Pi signatures:

```text
--kv-harness-prefixes my-agent="You are My Agent," \
--kv-harness-prefixes another-agent="You are Another Agent."
```

Matching ignores leading whitespace and letter case. OpenAI text content parts are
joined before matching.

`--lazy-restore on` is the default. Set it to `off` for the eager parity baseline.

The default budget is zero, which disables all disk-prefix work. The byte budget applies to
all complete entry files in the directory. Files are evicted by oldest last-use time.

Version 3 stores one complete prefix, a page boundary index, and the required KV dtype tag per
entry. Older formats are counted as stale and are not restored. The key combines the FTW
fingerprint, a hash of the runtime model geometry, TP rank and size, and the exact token chain.
Restore still compares the stored token tensor with the request prefix, so the digest is never
trusted as proof of equality. A startup scan reads safetensors headers only. Foreign fingerprints
are skipped, incomplete temp files are removed, and corrupt entries are deleted without failing
a request.

Each payload contains:

* `token_ids`: the verified prefix token chain
* `qsa_kv`: compact K and V rows for the 12 QSA layers
* `qsa_block_index`: token boundaries for page-granular QSA KV reads
* `qsa_index`: compressed QSA index rows, one per four tokens
* `conv` and `recurrent`: the 36 GDN layers at the prefix boundary
* `slot_state.*`: PLE convolution and n-gram state declared by the model config
* `qsa_pending`: the request-local QSA carry ring from the shared MTP state model

Restore eagerly installs GDN, PLE, QSA carry, the compressed QSA index, the sink page, and the
newest QSA-budget-sized run of KV pages. Decode can then begin while a background reader installs
the remaining KV pages newest-first. A QSA selection that reaches an absent page temporarily uses
the eager execution path, installs that complete page synchronously, and only then launches the
paged attention gather. Page publication uses an absent/loading/resident state machine, so no
reader can observe a partially copied page. CUDA graph replay resumes after all pages are resident.
This explicit presence bitmap was chosen over UFFD because QSA already exposes the selected
logical token indices immediately before its paged gather. The check stays at KV-page granularity
and does not require changing the shared MoE pager.

Writes first stage immutable host tensors on the scheduler stream. A bounded background queue
does the safetensors write, file sync, atomic rename, and LRU pass. A full queue drops the new
write and increments `write_drops`; write-side disk I/O never runs in the decode loop. A selected
missing KV page can still perform the intended synchronous read on the demand-fault path.

Configured coding-harness requests can also materialize the stable system-and-tools root as its
own entry. The tokenizer recognizes a configured system prompt signature, renders the leading
system run with the same tool schemas and template arguments, and takes the exact token common
prefix with the full prompt. The boundary is rounded down to the hybrid recurrence alignment.
If the template requires a user turn, two distinct probe queries supply that turn;
only their shared token prefix is compared with the real request. Probe queries
are never sent to the model, and a failed probe leaves normal tokenization intact.

The root entry is written only when all of these conditions hold:

* a nonzero disk-prefix budget created a `DiskPrefixStore` for a hybrid radix cache
* the aligned anchor lies strictly inside the current prefill chunk, including a
  single or final chunk
* the anchor is also aligned to the disk cache page size
* the request still owns a valid table row and the bounded writer accepts the job

The scheduler stages that snapshot directly to disk. It never inserts the harness root into the
live radix tree and never changes KV page or recurrent-slot ownership. The live tree therefore
keeps exactly the same deepest checkpoint it would keep for a prompt with no harness match.
For a final chunk, the unused request-owned ping-pong slot holds the root while
the usual slot holds the deepest continuation checkpoint. The scheduler stages
the root before donating or freeing either slot; no extra prefill is needed.
Anchors exactly at the chunk edge, disabled disk storage, unaligned anchors,
and unknown clients retain the normal cache behavior. A later session whose first user
message differs can restore a successfully written root, including after restart.

Scheduler status lines expose `harness_anchor_persisted`, its
`harness_anchor_persisted_intermediate` and `harness_anchor_persisted_final` breakdown,
`harness_anchor_skipped_final_chunk`, `harness_anchor_skipped_no_store`, and
`harness_anchor_skipped_unaligned` alongside the other disk-prefix counters.

### Single-chunk restart validation on node-4

On 2026-09-22, revision `06f30d8` passed a real serving check with the Qwen
Flash 4090 profile, 8192-token prefill chunks, 100352 reserved KV tokens,
a fresh 2 GiB disk-prefix directory, and eager restore (`--lazy-restore off`).
Each phase used a fresh server process. A 4561-token OpenCode-style system/user
request created a 4416-token shared root and the normal 4544-token continuation
checkpoint. The final-anchor persistence counter incremented once.

After restart, a different user query sharing only the system prefix restored
4416 tokens, leaving 145 to prefill. The cache reported one hit, 173,329,400
bytes restored and 31.28 ms of eager restore work. The same second query was
then run with a fresh empty cache:

| Second query | Cached tokens | First text | Request wall | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Restored shared root | 4416 | 7.33 s | 14.14 s | 81 |
| Empty cache | 0 | 14.40 s | 26.61 s | 81 |

The restored and cold requests had identical request hashes, complete answer
bytes and output counts, and both passed the ordered JSON-copy check. This
proves root reuse across the tested restart and preserves the deeper saved
checkpoint; it is one narrow fidelity/timing sample, not broad quality
equivalence or a new chunk-size qualification. Separate Linux scheduler/cache
regressions and CUDA GDN/PLE snapshot parity checks cover snapshot ownership
and state correctness. The tokenizer's 22 targeted tests also passed on Linux,
and the actual model tokenizer detected a stable root for different user queries.

Private artifacts are under
`node-4:/var/lib/longhorn/nvme-02/freetoken/results/prefill-depth-20260922/`,
using `root2-*.json`, matching journals/commands and `root2-cache/`. The earlier
failed `root-*` run is retained: it exposed the template's rejection of a
system-only conversation, which the two-query fallback addresses. The controller
restored the original serving configuration after validation.

### Final-chunk restart validation at the selected chunk size

The same revision and fixture also passed with the selected 2048-token chunk
size, placing the 4416-token root inside the final chunk after two full chunks.
The seed retained both the shared root and the normal 4544-token continuation
checkpoint. Each phase again used a fresh server process and eager restore.

| Second query | Cached tokens | First text | Request wall | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Restored shared root | 4416 | 7.05 s | 12.89 s | 81 |
| Empty cache | 0 | 22.11 s | 34.87 s | 81 |

The requests, answer bytes and output counts matched exactly. All three phases
passed the ordered JSON-copy check. First text was 68% faster and request wall
time was 63% lower for this single restored/cold comparison. These are narrow
fixture results, not a broad decode or quality qualification. Artifacts use
the `root3-*` prefix in the same private results directory. The original serving
configuration was restored after the test. A separate offline check using the
actual model tokenizer and a tool schema also found identical shared-root tokens
across different user queries.

### Restart validation with lazy restore enabled

The same 2048-token final-chunk check passed at revision `06f30d8` with
`--lazy-restore on`, matching the selected serving mode. The restored second
query reused 4416 tokens after restart, with first text in 6.98 s and request
wall time of 12.86 s. The identical query against an empty cache took 22.17 s
to first text and 32.26 s total. All three phases passed; restored and cold
requests, answer bytes and 81-token output counts matched exactly.

The restore reported one hit, 173,329,400 bytes restored, 36 streamed blocks,
zero faulted blocks and 50.06 ms of eager restore work. This exercises the
configured lazy path but does not establish coverage of every fault-on-demand
case. No corruption, fingerprint mismatch or dropped write was reported.
Artifacts use `root4-*` in the same private directory. The bounded controller
completed successfully and restored the original serving configuration.

### Larger-chunk candidate with lazy restore

The experimental fault-policy revision `39f5dc2` also passed the same three-phase
restart check with 8192-token chunks, fixed adaptation interval 1000 and lazy
restore enabled. Each phase used a fresh server; seed and restore shared a
private 2 GiB disk-prefix directory, while the cold phase used an empty one.

| Second query | Cached tokens | First text | Request wall |
| --- | ---: | ---: | ---: |
| Restored shared root | 4416 | 7.34 s | 16.89 s |
| Empty cache | 0 | 10.96 s | 22.15 s |

All three phases passed. Restored and cold requests, complete text, reasoning,
finish reasons and prompt/output counts matched exactly. The restore reported
36 streamed blocks, zero faulted blocks and 59.31 ms of eager work. As with the
earlier lazy check, this does not cover every fault-on-demand case. The bounded
controller completed and restored serving; artifacts use `root5-*` in the same
private directory. This validates prefix persistence for that candidate, not its
overall performance: the fixed-interval profile failed the separate continuation
performance comparison and remains unselected.

### Matched continuation check of the merged runtime

On 2026-09-22, baseline `4fbc4eb` and merged candidate `11654ed` each ran
three three-turn conversations with the existing `fixed-continuation-wall.py`
protocol. Each arm started a fresh server with 2048-token chunks, 100352
reserved KV tokens, lazy restore enabled, and its own empty 2 GiB disk-prefix
directory. Native kernels and all performance settings were unchanged.

| Runtime | Warm-up conversation | Measured conversation 2 | Measured conversation 3 | Measured mean |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 96.23 s | 77.83 s | 71.14 s | 74.49 s |
| Merged root fix | 98.68 s | 77.78 s | 70.77 s | 74.27 s |

All 18 responses passed. The protocol's `fixed_work_mismatches` check found
identical requests, complete answer messages, finish reasons, prompt counts
and output counts across arms. The measured mean differed by 0.3%, which is
consistent with preserved continuation performance in this small sample, not
evidence of a general speedup or broad model quality equivalence. This ordinary
continuation workload complements the shared-root restart checks above.

Artifacts use `rootqual1-*` in the same private results directory. The per-arm
`.revision` files identify each runtime; the common metadata file records the
baseline revision. The controller completed successfully and restored the
original service configuration before the next scheduled experiment.

For the RadixArk Qwen3.8 Flash-Next geometry at TP=1 and bf16, a 32,768-token entry is about
902.4 MiB before its small safetensors header:

| Component | Calculation | Size |
| --- | --- | ---: |
| QSA K/V | 32,768 x 12 layers x 2 K/V x 2 heads x 256 x 2 bytes | 768 MiB |
| QSA compressed index | 32,768 x 12 x 128 x 2 bytes / 4 | 24 MiB |
| GDN convolution | 36 x 10,240 x 3 x 2 bytes | 2.11 MiB |
| GDN recurrent | 36 x 48 x 128 x 128 x 4 bytes | 108 MiB |
| PLE state | one 10,240 x 9 bf16 convolution state plus two int32 IDs | 0.18 MiB |
| QSA carry and token IDs | 12 x 4 x 128 bf16 plus 32,768 int32 IDs | 0.14 MiB |

At the measured 116 token/s prefill rate, recomputing 32,768 tokens costs about 282 seconds.
Reading and installing about 0.88 GiB should be dominated by sequential NVMe read and host to
device transfer, roughly 0.3 to 1.0 seconds on the target class of machine. This is an estimate,
not a benchmark. Runtime logs report `restore_eager_ms`, `blocks_faulted`, `blocks_streamed`, and
`first_token_after_restore_ms`, as well as total `restore_ms` and estimated prefill time saved.

Chunked or delta-encoded storage is intentionally deferred. Whole-prefix granularity keeps the
first format simple and makes atomic replacement, validation, and deletion straightforward.
