# Direct file reads for selected DISK prefill

This experiment tests whether bypassing page-cache insertion during staged
prefill leaves more useful RAM residency for subsequent CPU decode. Serving
keeps buffered reads by default. The cache-aware follow-up can be selected
with `--moe-disk-prefill staged --moe-disk-prefill-io cached`. It requires
Linux ordinary file banks supporting direct I/O and native NVFP4 CUDA staging
with CPU decode. The staged token threshold still applies. There is no
automatic policy change.

`--moe-disk-prefill-io buffered` retains the existing reader. The unsuccessful
direct-only reader remains available through the constructor for ablations;
it is not a CLI policy. Startup logs identify the selected file I/O policy,
token threshold, and fixed ring allocation, without adding request telemetry.

The reader opens each ordinary file bank with `O_DIRECT`. It aligns a view
inside each existing pinned allocation, reducing usable chunk capacity when
needed while retaining the governor's fixed 64 MiB total allocation. Reads
round outward to 4 KiB blocks, including unaligned payload heads and tails.
Only the exact selected row bytes reach GPU scratch. Published HOT reuse,
router selections, scale bytes, and GPU arithmetic are unchanged.

The implementation reuses the loader's aligned short-read helper. EOF before
all requested payload bytes is an error. Application-level errors propagate
without buffered fallback. Buffer completion events retain the existing DMA
lifetime contract.

Linux documents filesystem-specific [direct-I/O alignment and concurrency
constraints](https://man7.org/linux/man-pages/man2/open.2.html). The target is
node-4's Linux 6.8 ext4 filesystem, using the existing loader's conservative
4 KiB alignment. This reader is created inside the serving worker and must
not run concurrently with a fork. Direct reads can sacrifice useful cache
hits and increase storage traffic, so correctness alone does not qualify
this as a throughput improvement.

The [ext4 read path](https://github.com/torvalds/linux/blob/v6.8/fs/ext4/file.c#L65)
can itself fall back to buffered I/O for some unsupported inode features.
Requesting `O_DIRECT` alone is therefore insufficient proof of transport.
During the direct start, `STATX_DIOALIGN` on all eight model files mapped by
the GPU worker reported 4-byte memory and 512-byte offset alignment, compatible
with this reader's 4 KiB requests. The experiment targets these checked files;
other filesystems and inode configurations are not qualified.
The [metadata record](../bench/results/4090-direct-io-file-alignment-20260905.json)
retains the worker PID, mapped paths, statx masks, and both alignment fields.

Validation must cover exact packed bytes, partial source views, unaligned
allocation addresses and file offsets, repeated asynchronous buffer reuse,
short reads, EOF, and real NVFP4 GEMM output bits. A file-residency check
compares buffered and direct reads using `mincore` on a private test file.
The performance gate uses whole client response time with diagnostics off,
matching memory geometry, full warmups, reversed starts, complete JSON,
and the existing long fidelity questions.

## Initial validation

At `2b6da47`, all 53 focused Linux CUDA checks in
`test_disk_prefill_staging.py` and `test_materialize_hot_slots.py` passed.
The same 53 checks passed under CUDA memcheck with zero errors. Direct and
buffered transports retain identical BF16 GEMM output bits at 16 and 512
tokens. The private-file residency check observed zero resident pages after
direct reads and all sixteen pages resident after buffered reads. This
confirms the intended cache behavior on the target filesystem, without
claiming a model speedup. The [log](../bench/results/4090-direct-io-validation-20260905.txt)
includes exact commands and verification of the restored original service.

## Cache-hit reuse candidates

If direct reads lose too much useful cache reuse, a later experiment could
read resident ranges from RAM and use direct I/O only for the rest. That is
a source-selection policy; it must still execute every selected expert.

`RWF_NOWAIT` is not by itself a cache-only read on the target kernel. Linux
6.8's [generic file read path](https://github.com/torvalds/linux/blob/v6.8/mm/filemap.c#L2558)
explicitly permits readahead with `IOCB_NOWAIT`, and its miss path can start
readahead before returning `EAGAIN`. A successful nonblocking API call is
therefore insufficient evidence that a policy avoids cache insertion.

The [mincore implementation](https://github.com/torvalds/linux/blob/v6.8/mm/mincore.c#L147)
provides a residency snapshot, but it can become stale immediately. It also
reports every page resident when ownership and permission checks disallow
exposing residency. Any use as a transport hint must preserve correct reads when
the hint is stale or unavailable, and must measure the cost of inspecting
pages and splitting transfers. The original direct-reader tests do not
performance-qualify these candidates.

## Whole-response result: direct-only reads do not qualify

Four isolated starts at runtime `2b6da47` compared buffered/direct/direct/buffered
transport. Both modes used staged GPU prefill with published HOT weight reuse,
automatic adaptation with unchanged phase aim and split histories, 20 PIN
layers, 28 DISK layers, and 82 HOT experts per DISK layer. Both allocated the
same 64 MiB ring. Diagnostics, GPU timing, HOT persistence, and KV reuse were
off. Full-response warmups preceded four measured requests per start.

| Mean client response time | Buffered | Direct | Result |
| --- | ---: | ---: | --- |
| 1,844-token prompt, 383-token JSON | 29.452 s | 30.869 s | 4.8% slower; 1/4 pairs faster |
| About 1,880-token prompt, 192-token prose | 22.702 s | 21.944 s | 3.3% shorter; 1/4 pairs faster |

The prose average benefits from one large improvement in the second direct
start. Six of eight measured pairs are slower. Total wall time for the fixed
eight-request mix increases from 208.615 to 211.249 seconds, 1.3% slower.
Host page cache is retained between starts, and the reversal changes the
aggregate direction. This does not establish a dependable general gain.

Direct I/O also increases mean whole-worker storage-read accounting from
5.137 to 27.178 GiB per JSON response and from 5.933 to 26.535 GiB per prose
response. These counters include other worker I/O, not just expert weights.
They are consistent with sacrificing useful buffered cache hits; they do
not independently attribute subsequent decode costs to particular evictions.

All twelve JSON responses, including warmups, pass value, integer-type,
key-order, and multiplicity checks, finish normally, and use 383 output tokens.
Every start scores 7/8 on the long fidelity questions, with identical answers
including the same code-trace failure (`108`, expected `68`). Prose remains
unscored. The transport's byte/GEMM tests establish exact arithmetic for the
same inputs; these model checks do not establish broad quality equivalence.

Keep buffered reads as the serving default. The direct-only constructor
option remains an experimental comparison point for a more selective reader,
with no CLI integration or default-policy change. The next candidate needs
to preserve useful RAM hits while reducing insertion of cold prefill data,
then repeat a complete-response gate.

The [complete record](../bench/results/4090-direct-io-wall-20260905.json)
retains all four starts, native binary identity, startup transport selection,
matching cache geometry, raw outputs, journals, exact clients and driver,
the measured transport source, I/O snapshots, and reproducible analysis.
All four measured JSON pairs have identical text; all eight pairs have
identical usage counts. The original service was restored and verified
with a real `OK` completion.

## Experimental reuse of resident rows

`DiskPrefillStaging(..., direct_io=True, reuse_cached_rows=True)` adds an
experimental source-selection policy. It inspects small groups of rows with
`mincore`, then uses buffered reads for fully resident rows and aligned direct
reads for the rest. Adjacent selected rows using the same source are coalesced.
The buffered descriptor requests random-access advice to limit readahead;
stale residency can still cause buffered reads to fetch missing pages. This
is an advisory performance policy, never permission to skip a required read.

The policy conservatively trusts residency only for files owned by the
serving user. Failed queries and other ownership use direct reads. It masks
the defined residency bit, handles partial source views and rows larger than
the probe window, and keeps all row data authoritative in the original file.
The 64 MiB pinned allocation is unchanged. On the target's 4 KiB pages, the
bitmap and bounded snapshot copies use about 32 KiB at peak, plus small
Python objects, independent of model size and request length.

The hint lookup and range planning are functional work included in wall time.
No diagnostic counters or device readbacks are added. Default serving remains
buffered; the `cached` CLI policy enables both constructor options.

At `25cdbff`, all 70 focused staging/materialization CUDA tests pass normally
and under GPU memcheck, with zero errors. Mixed-residency cases check the
actual descriptor flags and prove that fully resident rows use buffered reads,
partially resident rows use direct reads, and initially cold pages remain
uncached. Other cases cover failed queries, concealed residency, stale hints,
undefined high residency bits, large rows, partial views, and exact NVFP4
GEMM output bits. The [validation log](../bench/results/4090-cached-io-validation-20260905.txt)
retains exact commands and restoration verification.

## Whole-response result: cache-aware reads improve this gate

Four isolated starts at runtime `25cdbff` compared
buffered/cached/cached/buffered transport. Both modes used the same staged
prefill implementation, published HOT reuse, native extension, memory
geometry, and automatic adaptation. Diagnostics, GPU timing, HOT persistence,
and KV reuse were off. Each start had two full-response warmups followed by
four measured responses. Residency inspection and range planning are included
in the client timer. This gate selected the constructor options through a
startup probe before CLI integration; it does not validate the later CLI path.

| Mean client response time | Buffered | Cache-aware | Wall-time reduction |
| --- | ---: | ---: | ---: |
| 1,844-token prompt, 383-token JSON | 29.872 s | 23.894 s | 20.0% |
| About 1,880-token prompt, 192-token prose | 22.587 s | 18.956 s | 16.1% |
| Fixed eight-request mix, total | 209.835 s | 171.400 s | 18.3% |

All eight matched responses are faster. The first matched start pair improves
by 11.4% and the reversed pair by 24.1%. Mean first-token time falls from
9.867 to 6.193 seconds for JSON and from 9.773 to 6.271 seconds for prose.
JSON decode falls from 20.005 to 17.702 seconds; prose decode is nearly flat
at 12.814 versus 12.685 seconds. This is primarily a prefill improvement, with
no measured average decode penalty in this gate.

Whole-worker storage-read accounting is 7.038 versus 6.174 GiB per JSON
response and 5.844 versus 6.115 GiB per prose response. Unlike direct-only
reads, the cache-aware mode avoids a large increase in this accounting.
These are whole-worker counters, not expert-only traffic or proof that
specific decode pages were protected from eviction.

All twelve JSON responses, including warmups, finish normally and pass strict
value, integer-type, key-order, and multiplicity checks. All four measured
JSON pairs have identical text; all eight timing pairs have identical usage.
Every start scores 7/8 on the long fidelity questions, with identical answers
including the existing `108` code-trace failure (expected `68`). Prose differs
and remains unscored. This is limited regression evidence, not broad quality
qualification.

The [complete record](../bench/results/4090-cached-io-wall-20260905.json)
contains raw outputs, exact clients and driver, measured transport source,
native identity, startup geometry, journals, I/O snapshots, and reproducible
analysis. Separate metadata inspection confirms that all eight mapped model
files are owned by the observed worker's effective user, allowing the
residency policy. The original service was restored and verified with a real
`OK` completion.

The result qualifies this finite comparison against buffered staging. Host
page cache is retained between starts, and each start has limited adaptation
history. The later combined comparison below covers the original CPU server;
the [sustained follow-up](sustained-prefill.md) measures a smaller gain and a
generation penalty. A general production claim remains unqualified.
This 18.3% cannot be added to earlier gains against different baselines.

## Serving configuration validation

At `78848ce`, `--moe-disk-prefill-io cached` is wired from the CLI through
engine configuration into the staging reader. Invalid file policies and
cached reads with CPU or copy prefill are rejected. The ordinary serving
defaults remain CPU prefill and buffered reads.

All 99 focused Linux CUDA checks in staging, materialization, and policy
validation pass normally and under GPU memcheck with zero errors. The actual
cache initialization path is exercised with both file policies, protected HOT
rows, overlap on/off, and fused-copy fallbacks. Both modes preserve exact
selected bytes and leave uncopied scratch rows unowned. The
[validation log](../bench/results/4090-cached-io-cli-validation-20260905.txt)
retains commands, revision, and a real completion from the restored original
service. The direct original-versus-combined model comparison uses this
frozen revision and the real CLI, without a constructor probe.

## Combined configuration versus the original CPU server

The completed four-start original/optimized/optimized/original comparison
uses original `3a67403` with CPU DISK prefill and combined `78848ce` with
`--moe-disk-prefill staged --moe-disk-prefill-io cached`. Both modes use the
same automatic adaptation, placement, and cache geometry. Diagnostics, GPU
timing, HOT persistence, and KV reuse are off. Exact native identities and
worker mappings are verified. This includes the preceding optimization
stack, so it does not isolate the contribution of the reader alone.

| Measured client response | Original CPU | Combined cached staging | Wall-time reduction |
| --- | ---: | ---: | ---: |
| Complete JSON, mean of four requests | 36.703 s | 23.765 s | 35.3% |
| Prose at a 192-token cap, mean of four requests | 28.192 s | 17.089 s | 39.4% |
| Fixed eight-request mix, total | 259.577 s | 163.413 s | 37.0% |

All eight pairs improve; the first matched start pair improves by 32.2% and
the reversed pair by 41.8%. Mean first-token time falls from about 15 seconds
to about 6 seconds. All twelve JSON responses finish normally and pass strict
value/type/order/multiplicity checks. One original JSON response uses extra
whitespace and 449 output tokens instead of 383. The seven timing pairs with
matching usage still improve by 35.5% (220.144 to 141.901 seconds). Three of
four measured JSON pairs have identical text.

All four starts score 7/8 on the long fidelity questions, with the same
code-trace failure. The first original start answers `100`; the other starts
answer `108`, against the correct `68`. Prose differs, reaches its output cap,
and remains unscored. These are limited model checks alongside exact byte
and GEMM tests, not broad quality qualification.

Whole-worker storage-read accounting is 5.669 versus 5.232 GiB per JSON
response and 5.114 versus 5.198 GiB per prose response. This comparison does
not measure expert-only traffic or establish particular page-eviction causes.
The [complete record](../bench/results/4090-combined-cached-wall-20260905.json)
retains the driver, clients, transport source, native identity, matching
geometry, raw outputs, journals, and reproducible analysis. The original
service was restored and verified with a real `OK` completion.

The [completed sustained follow-up](sustained-prefill.md) at `132b629`, with
unchanged inference code, measures 17.8% less wall time across 24 responses
per policy (634.481 to 521.738 seconds). Complete prose replaces the earlier
capped task. The source-balanced gain falls from 23.1% in the first half to
11.4% in the second half, and generation after the first token is slower on
average. The earlier 37.0% result applies to its short sequence. Sustained
reader isolation, cache admission, and broader quality remain open.
