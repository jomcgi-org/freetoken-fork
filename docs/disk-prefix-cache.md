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

The root entry is written only when all of these conditions hold:

* a nonzero disk-prefix budget created a `DiskPrefixStore` for a hybrid radix cache
* the request is split across multiple prefill chunks
* the aligned anchor lies strictly inside the current non-final chunk
* the anchor is also aligned to the disk cache page size
* the request still owns a valid table row and the bounded writer accepts the job

The scheduler stages that snapshot directly to disk. It never inserts the harness root into the
live radix tree and never changes KV page or recurrent-slot ownership. The live tree therefore
keeps exactly the same deepest checkpoint it would keep for a prompt with no harness match.
Single-chunk prompts, anchors reached only by the final chunk, disabled disk storage, unaligned
anchors, and unknown clients retain the normal cache behavior. A later session whose first user
message differs can restore a successfully written root, including after restart.

Scheduler status lines expose `harness_anchor_persisted`,
`harness_anchor_skipped_final_chunk`, `harness_anchor_skipped_no_store`, and
`harness_anchor_skipped_unaligned` alongside the other disk-prefix counters.

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
