# Concurrent CPU and GPU prefill

For mixed native NVFP4 HOT/COLD prefills, this opt-in schedule stages the CPU
inputs before launching the GPU partial:

```sh
FREETOKEN_PREFILL_HOT_OVERLAP=1 ft serve <normal arguments>
```

The default is `0`, which preserves the serial schedule. HOT-only and COLD-only
route sets keep their existing paths. The switch does not change decode.

## Scheduling and correctness

The serial path launches HOT GPU work, then makes blocking CPU input copies.
Those copies wait for earlier GPU work, so the CPU kernel starts afterward.
The concurrent path prepares the cold expert lease and copies activations,
expert IDs, and weights to the existing pinned buffers first. A callback then
launches the unchanged GPU partial, followed by the existing synchronous native
CPU kernel. CUDA executes the GPU work while the host runs that CPU kernel.

There are no extra threads, streams, or persistent buffers. CPU output returns
on the original CUDA stream, which orders it after the GPU partial. The final
addition retains `gpu_routed + cpu_routed`. Router output, HOT placement, packed
weights, scales, quantization, and each partial's arithmetic stay the same.

The callback runs outside the CPU batch degradation handler. GPU allocation
failure therefore does not disable the CPU batch kernel. The cold lease is
released before the existing full-CPU fallback stages every selected route.
Unsupported grouped GPU geometry retains its existing decode-kernel fallback.

## Validation

The focused checks run on node-4 with its existing native extensions:

```sh
python -m pytest \
  tests/moe/test_prefill_hot_overlap.py \
  tests/moe/test_disk_tier.py \
  tests/moe/test_cpu_moe_prefill_batch.py -q
```

A diagnostic CUDA test confirms that the CPU kernel starts before the GPU
completion event becomes ready, and that returned output is ready to consume.
Native NVFP4 tests compare both schedules bit for bit, including Qwen's
H=2560/I=640 expert geometry, repeated experts, all-cold token rows, and repeated
workspace reuse. Separate tests check copied input snapshots, callback errors,
CPU batch degradation, route partitioning, lease cleanup, and GPU fallbacks.

## Non-debug wall-time comparison

Reuse `bench/hot-routing-wall.py` for identical prompts and alternating mode
order. In this experiment its `parallel` field means concurrent CPU/GPU prefill.
The benchmark server reads the same eight-byte control mmap at each prefill
boundary and assigns its boolean value to
`freetoken.layers.moe._PREFILL_HOT_OVERLAP`. The hook collects no data.

Keep MoE diagnostic collection and GPU timing disabled in both modes, and do
not poll profile endpoints. Hold HOT placement fixed with adaptation and plan
persistence disabled. Use the naive KV cache and disable disk KV reuse so
paired identical prompts cannot reuse cached prefixes. Keep selective prefill
transfers disabled and parallel HOT routing enabled in both modes.

These settings isolate the scheduling change. They are benchmark controls,
not production recommendations. Judge the change by client first-token and
whole-request wall time, excluding warmups, and verify text and usage equality
for every matched pair.
