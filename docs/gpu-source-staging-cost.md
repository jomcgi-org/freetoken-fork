# GPU source staging cost

The existing GPU-fetch transport copies selected file-backed expert rows into a
bounded pinned ring before gathering them into the GPU cache. Direct GPU layers
instead retain full pinned host banks and gather from those addresses. Before
changing host placement, measure the extra transfer cost that making those
full banks reclaimable would introduce.

`bench/gpu-source-staging-cost.py` compares the existing production
`OffloadMoeCache.copy_missing` paths with synthetic Qwen Flash NVFP4 banks
(H=2560, I=640, 512 experts). It copies packed weights, block scales and global
scales. Each mode receives identical GPU-resident miss IDs and destination
slots. Both use CUDA graphs and fused copies; staging includes its actual D2H
control copies, native coordinator and H2D gather.

The source working set exceeds the CPU's last-level cache. Files are flushed
and mapped pages populated before timing. Captured row rotation visits all 512
experts, avoiding repeated copies from one small cached row. Miss counts cover
zero, sparse misses, and up to forty distinct routes. Paired repeats alternate
order. Wall time includes graph submission and completion waiting, with no new
GPU timing events. Byte checks and native work-count checks run outside timing.
Every copied byte and untouched destination row is checked, including zero-work
replays. Failed attempts and partial records are retained.

Run only after other model benchmarks have ended, under an exclusive detached
supervisor with a tested original-serving recovery hook. The script refuses an
occupied GPU and does not stop serving itself. Its runtime imports must point
at a clean, qualified CUDA checkout; it records source and native identities.

```sh
python bench/gpu-source-staging-cost.py \
  --scratch-dir /path/to/private/scratch \
  --output /path/to/private/fresh-result.json
```

This measures resident transfer cost. It does not include expert selection,
model arithmetic, cache replacement, cold storage, RAM pressure or the benefit
of reclaiming GPU duplicates. The current global GPU-fetch mode also changes
the CPU/GPU compute split, so its model throughput is a separate question. A
future placement experiment must preserve that split and existing arithmetic,
then qualify complete-response wall time.

CUDA execution has passed the byte and native-work assertions across every
declared miss count. Original serving recovery was also verified. These checks
qualify the probe; model placement remains a separate experiment.
