# Spec: disk-backed prefix state cache (FreeToken, patch 18, "LMCache lane")

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (current tip).
Commit on the branch; do NOT push. Study first: the radix prefix cache
(cache_type=radix, its lookup/insert/evict seams), the hybrid
KV/mamba-slot layout (12 QSA layers keep paged KV; 36 GDN layers keep
fixed-size recurrent state per sequence), and the MTP K=1 exact
snapshot/restore machinery in engine/engine.py + spec_decode.py (it
already captures GDN/QSA/PLE per-request state; that is the state model
to reuse).

## Problem (measured, node-4)

Warm prefill is 116 tok/s: a 32k-token agent context costs ~4.5 minutes
to ingest, every time it falls out of the VRAM radix cache or the server
restarts. The agent-factory workload replays the same large system
prompts and repo contexts all day. This model is the BEST case for disk
KV: 36 of 48 layers are GDN with fixed-size state instead of per-token
KV, so a whole prefix persists as (12 QSA layers' KV pages for L tokens
+ 36 fixed GDN states at position L) - a fraction of a full-attention
model's footprint. The box has 1.5TB of free NVMe.

## Task

1. Flags: --kv-disk-cache-dir PATH and --kv-disk-cache-gib N (default 0
   = off). Byte-budgeted LRU over the store (delete oldest by last-use;
   an index file or per-entry mtime both fine - simple and crash-safe
   beats clever).
2. Persist: when a request completes (or its radix node is about to be
   evicted), asynchronously write the prefix state: token-id chain hash
   as the key, payload = QSA KV pages + GDN/conv/mamba states at the
   prefix boundary + metadata (token ids for verification, lengths,
   model fingerprint). Never block the decode loop on writes
   (background thread, bounded queue, drop-on-overflow with a counter).
3. Restore: on a new request whose VRAM radix match is shorter than a
   stored prefix, load the LONGEST stored entry whose token ids
   prefix-match the request exactly (verify against stored token ids,
   never trust the hash alone), install KV pages + states, set
   cached_len, and let normal prefill continue from there.
   Whole-prefix granularity is fine for v1; chunked storage is future
   work, note it.
4. Correctness bar: greedy continuation after a disk restore must be
   bit-identical to full recompute (the MTP snapshot/restore parity
   test pattern applies). Key everything by the FTW fingerprint + model
   config hash: a store written by a different checkpoint must never
   load (silently skip, count it).
5. Crash safety: partial writes must never restore (write temp +
   rename). A corrupt entry is deleted and counted, never fatal.
6. Stats on the decode/prefill log: disk_prefix hits/misses,
   bytes_restored, restore_ms, and estimated prefill_ms_saved
   (tokens_restored / measured prefill rate).

## Tests

GPU-free: store round-trip with synthetic states, LRU budget eviction,
fingerprint mismatch rejection, torn-write recovery, token-id
verification. CUDA-gated: bit-identical continuation after restore.
Platform note: the Mac cannot run the package's tests (linux-only
deps); write them, state that plainly, never fake a pytest line.

## Deliverable

Commits on the branch + report: files, state payload layout + size math
for a 32k prefix on this model, restore-time estimate vs prefill cost,
deviations.
