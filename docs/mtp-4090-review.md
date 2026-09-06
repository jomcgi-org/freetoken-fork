# MTP follow-up for the 4090

This is a source and checkpoint review, not a performance result. It examines
the user's `feat/mtp-graphs` branch at
[`1c42624abf254aca83659badb99fa194f63f4d6a`](https://github.com/jomcgi-org/freetoken-fork/tree/1c42624abf254aca83659badb99fa194f63f4d6a).
The controlled Pi comparison continues with MTP disabled and unchanged sources.

## The deployed checkpoint has no draft head

A read of `models/flash-e2m1.ftw/freetoken_weight.json` on node-4 on
2026-09-06 found 794 `weight` entries, 288 `experts_bank` entries, and no `mtp`
entries. Its explicit counts also report `mtp: 0`, with `mtp_quant: null` and
`mtp_expert_bytes: 0`. The 187153-byte index has SHA-256
`6006a912bcc1075ffab1cb590b8251104ad0c6bec6726a930b4038e9fb4249b4`.
Only the JSON index was read; no tensor shards were scanned or converted.

The [loader](https://github.com/jomcgi-org/freetoken-fork/blob/1c42624abf254aca83659badb99fa194f63f4d6a/python/freetoken/engine/engine.py#L601)
rejects `--speculative-mtp on` for an FTW checkpoint without those entries.
An MTP-equipped checkpoint is therefore a prerequisite for a model test. Preserve
the existing target tensor bytes and qualify the draft-head addition separately.

## Costs that graph capture does not remove

The [verification path](https://github.com/jomcgi-org/freetoken-fork/blob/1c42624abf254aca83659badb99fa194f63f4d6a/python/freetoken/engine/engine.py#L1653)
creates and records six CUDA timing events per MTP step for draft, snapshot, and
verification. These calls have no diagnostic flag guard. The scheduler resolves
all three elapsed times after the output fence. Gate these events behind an
explicit timing option before measuring non-debug MTP wall time. This observation
does not assign a percentage to their overhead, and they are not executed in the
current MTP-disabled Pi comparison.

The [PLE staging helper](https://github.com/jomcgi-org/freetoken-fork/blob/1c42624abf254aca83659badb99fa194f63f4d6a/python/freetoken/spec_decode.py#L33)
copies the draft token to pinned host memory and synchronizes that copy before
uring staging and graph replay. The verifier also snapshots recurrent, convolution,
and QSA pending state. On rejection it restores that state and recomputes the seed
with an eager width-one target forward. These costs remain outside or alongside
the captured width-two target computation.

The [resident draft head](https://github.com/jomcgi-org/freetoken-fork/blob/1c42624abf254aca83659badb99fa194f63f4d6a/python/freetoken/models/qwen4_exp/mtp.py#L34)
resets its attention history for each one-token draft. It does not maintain the
prompt's independent draft-head KV history. Lower acceptance is a possible cost
of that approximation; the target verifier is responsible for output correctness.
The head also consumes GPU memory before the target expert-cache budget is set.
The branch documents 4.6875 GiB for its BF16 experts or 1.3220 GiB for NVFP4
experts, plus the remaining BF16 head tensors. A future comparison must report
the resulting target expert slots and storage traffic as well as acceptance.

## Qualification before adoption

For one draft per step, the useful cost condition is approximately
`draft + snapshot + verify + (1 - acceptance) * rejection_replay <
(1 + acceptance) * ordinary_decode`, using costs at the actual memory geometry.
Acceptance alone cannot establish a speedup. This expression omits request-end
effects and includes all staging and launch costs within their respective phases.

First check accepted and rejected windows against ordinary greedy decoding,
including recurrent/PLE/QSA state, prefix reuse, EOS, and output limits. The branch's
helper and graph tests are useful but do not by themselves establish whole-model
equivalence. Then compare MTP off, MTP with eager verification, and MTP with graph
verification under repeated start orders, fixed requests, optional diagnostics off,
and complete client wall time. Keep draft precision separate from target precision.

The branch's `MTP-GRAPHS-VERIFY.md` records earlier eager MTP timings, but their raw
runs were not revalidated in this review. No MTP throughput gain is claimed here.
