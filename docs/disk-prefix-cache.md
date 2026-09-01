# Disk-backed hybrid prefix cache

Enable the lane with both flags:

```text
--kv-disk-cache-dir /nvme/freetoken-prefixes --kv-disk-cache-gib 1024
```

The default budget is zero, which disables all disk-prefix work. The byte budget applies to
all complete entry files in the directory. Files are evicted by oldest last-use time.

Version 1 stores one complete prefix per entry. The key combines the FTW fingerprint, a hash
of the runtime model geometry, TP rank and size, and the exact token chain. Restore still
compares the stored token tensor with the request prefix, so the digest is never trusted as
proof of equality. A startup scan reads safetensors headers only. Foreign fingerprints are
skipped, incomplete temp files are removed, and corrupt entries are deleted without failing a
request.

Each payload contains:

* `token_ids`: the verified prefix token chain
* `qsa_kv`: compact K and V rows for the 12 QSA layers
* `qsa_index`: compressed QSA index rows, one per four tokens
* `conv` and `recurrent`: the 36 GDN layers at the prefix boundary
* `slot_state.*`: PLE convolution and n-gram state declared by the model config
* `qsa_pending`: the request-local QSA carry ring from the shared MTP state model

Writes first stage immutable host tensors on the scheduler stream. A bounded background queue
does the safetensors write, file sync, atomic rename, and LRU pass. A full queue drops the new
write and increments `write_drops`; disk I/O never runs in the decode loop.

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
not a benchmark. Runtime logs report measured `restore_ms` and estimate saved prefill time from
the observed prefill rate.

Chunked or delta-encoded storage is intentionally deferred. Whole-prefix granularity keeps the
first format simple and makes atomic replacement, validation, and deletion straightforward.
