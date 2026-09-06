# File backing for GPU expert sources

`--moe-gpu-source staged` is an opt-in native NVFP4 placement experiment. It
resolves the ordinary GPU, CPU and HOT layer selections first, then keeps the
GPU layers' host banks in their existing checkpoint files. Decode cache misses
copy exact packed weights, block scales and global scales through the existing
bounded GPU-fetch ring. The default remains `pinned`.

CPU and HOT layer selections, expert routing and GPU arithmetic stay on their
existing paths. GPU-source prefill always uses the GPU, including chunks below
the ordinary DISK prefill crossover. CPU/DISK layers retain that crossover.
Session advice continues warming GPU slots for the selected GPU layers.

The initial implementation requires the standard FTW or supported bank-index
loader, `--moe-backend offload`, native NVFP4 banks, `--moe-disk-prefill staged`,
`--moe-disk-decode cpu`, and `--moe-disk-pager madvise`.

```sh
ft serve --model /path/to/model \
  --moe-backend offload \
  --moe-cache-auto \
  --moe-disk-prefill staged \
  --moe-disk-decode cpu \
  --moe-disk-pager madvise \
  --moe-gpu-source staged
```

File backing makes pages reclaimable; it does not remove the need to read
expert weights. Each decode miss adds a CPU copy before the GPU transfer, and
file-backed prefill can lose overlap with preceding computation. Keep cache,
KV and compute-layer geometry fixed when comparing complete-response wall
time. Measure actual RAM use and storage reads along with that wall time.
Detailed timing diagnostics should remain disabled for the comparison.

Runtime and model validation are pending. Do not promote the option based only
on reclaimed pinned bytes or a resident transport benchmark.
