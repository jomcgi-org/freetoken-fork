# Prefill chunk telemetry

With `--moe-collect-stats`, `/v1/moe-layer-profile` includes a `prefill`
object containing the latest 256 completed chunks. Deduplicate by `sequence`
within a server instance. Polling does not consume the buffer.

Each chunk reports new `tokens`, `elapsed_ms`, `tokens_per_second`, and a
`requests` list with request uid, newly processed tokens and the completed prompt
position (including a cached prefix). A batched rate is aggregate, not per request.

The elapsed interval uses CUDA events around the prefill forward and sampled
output copy. It includes weight transfers, CPU waits and host dispatch gaps on
that stream. The scheduler resolves events after its existing completion fence;
there is no new synchronization and delayed polling cannot change the rate.
Scheduling, prefix restoration before the forward, tokenization, queueing and
transport are outside this interval. Keep client time to first token separate.

`dispatched_at_s`, `observed_at_s` and snapshot `clock_s` use the server monotonic
clock. Observation can lag completion because scheduling overlaps. Neither host
timestamp is an exact CUDA event timestamp. Plot completed chunk measurements,
not an interpolated claim of token-by-token progress. A one-chunk prompt yields
one measurement. These counters do not change the legacy `/v1/stats` rates.
