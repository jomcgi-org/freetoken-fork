# Hybrid prefix expert profiles

Final prefill can publish a reusable hybrid snapshot before decoding starts.
If the request finishes between page boundaries, its live state cannot be
attached to a shorter cache key. No new snapshot is inserted at that finish.
Previously, the collected expert-prefetch profile was then discarded even
though the earlier prefix remained cached.

On successful finish, retain the profile on the request's existing nonempty
cache handle before releasing it. Newly inserted snapshots still receive the
profile through the existing insertion path. Failed forwards do not publish
profiles, and an absent export does not erase an existing profile. The empty
root never receives session advice.

This retains advisory weight-placement metadata already collected by session
prefetching. It adds no profile collection, GPU copy, model arithmetic or
expert-routing change. KV pages and recurrent, convolution and PLE states
remain attached to their original token boundaries. No wall-time improvement
is claimed until a paired model benchmark qualifies the change.

Targeted state and ownership coverage is in
`tests/scheduler/test_prefill_snapshot_carry.py`. It checks short and longer
final chunks, aligned and unaligned finishes, failed forwards, absent exports,
empty prefixes and subsequent profile admission.
