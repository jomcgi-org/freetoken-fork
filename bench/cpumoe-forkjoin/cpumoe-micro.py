"""CPU cold-expert decode microbenchmark (nvfp4 W4A8, Qwen flash geometry).

Usage: cpumoe-micro.py BACKING THREADS COLD [TOKENS]
  BACKING  anon4k | thp | file   (file = page-cache mmap of a bank file, like DISK layers)
  THREADS  worker count; pinned like the serve pool (physical cores, then SMT siblings)
  COLD     cold experts per layer per token (serving averages ~3-4 of top-10)

Drives the real CpuMoeExecutor: one persistent decode task per layer (group_routes,
as serving uses), 29 layers per token, random distinct experts from 64 per layer (5 GB
working set, far above the 96 MB L3). Prints one JSON line: per-token wall, and from the
executor's own task timing the wake / compute split and compute bandwidth.
"""
import ctypes, json, mmap, os, sys, time
import torch
from freetoken.kernel import _cpu_moe
from freetoken.moe.cpu_executor import _ACT_IDS, _WFMT_IDS, resolve_threads_and_affinity

backing, threads, cold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 300
L, E, K, H, I = 29, int(os.environ.get("MICRO_E", "64")), 10, 2560, 640
FILE = "/var/lib/longhorn/nvme-02/freetoken/tmp/cpumoe-micro-bank.bin"

shapes = dict(gup=(E, 2 * I, H // 2), gus=(E, 2 * I, H // 16), gug=(E, 2 * I),
              dnp=(E, H, I // 2), dns=(E, H, I // 16), dng=(E, H))
dtypes = dict(gup=torch.uint8, gus=torch.uint8, gug=torch.float16,
              dnp=torch.uint8, dns=torch.uint8, dng=torch.float16)
def nbytes(k):
    n = 1
    for d in shapes[k]: n *= d
    return n * (2 if dtypes[k] == torch.float16 else 1)
ALIGN = 1 << 21
per_layer = sum((nbytes(k) + ALIGN - 1) // ALIGN * ALIGN for k in shapes)
total = per_layer * L

libc = ctypes.CDLL("libc.so.6", use_errno=True)
MADV_HUGEPAGE, MADV_NOHUGEPAGE = 14, 15
if backing == "file":
    if not os.path.exists(FILE) or os.path.getsize(FILE) != total:
        g = torch.Generator().manual_seed(0)
        with open(FILE, "wb") as f:
            left = total
            while left:
                n = min(left, 1 << 28)
                f.write(torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g).numpy().tobytes())
                left -= n
    fd = os.open(FILE, os.O_RDWR)
    region = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    # Warm the page cache (serving's DISK banks are mostly page-cache resident).
    with open(FILE, "rb") as f:
        while f.read(1 << 26): pass
else:
    region = mmap.mmap(-1, total, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    addr = ctypes.c_void_p.from_buffer(region)
    libc.madvise(ctypes.c_void_p(ctypes.addressof(addr)), ctypes.c_size_t(total),
                 MADV_HUGEPAGE if backing == "thp" else MADV_NOHUGEPAGE)
    del addr
    g = torch.Generator().manual_seed(0)
    view = torch.frombuffer(region, dtype=torch.uint8)
    step = 1 << 28
    for o in range(0, total, step):
        n = min(step, total - o)
        view[o:o + n] = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
    del view

base = torch.frombuffer(region, dtype=torch.uint8)
banks = {k: [] for k in shapes}
off = 0
for l in range(L):
    for k in shapes:
        n = nbytes(k)
        t = base[off:off + n].view(dtypes[k]).view(shapes[k])
        if k in ("gus", "dns"):
            t.bitwise_and_(0x07).bitwise_or_(0x30) if backing != "file" else None  # sane e4m3
        if k in ("gug", "dng") and backing != "file":
            t.fill_(0.01)
        banks[k].append(t)
        off += (n + ALIGN - 1) // ALIGN * ALIGN
tables = {k: torch.tensor([t.data_ptr() for t in banks[k]], dtype=torch.int64) for k in shapes}

n, cores = resolve_threads_and_affinity(threads)
ex = _cpu_moe.CpuMoeExecutor(
    num_threads=n, num_layers=L, num_experts=E, top_k=K, hidden_size=H, inter_size=I,
    max_tokens=4, activation_id=_ACT_IDS["silu"], apply_router_weight_on_input=0,
    weight_format=_WFMT_IDS["nvfp4"],
    gate_up_ptr=tables["gup"].data_ptr(), down_ptr=tables["dnp"].data_ptr(),
    gate_up_scale_ptr=tables["gus"].data_ptr(), gate_up_global_ptr=tables["gug"].data_ptr(),
    down_scale_ptr=tables["dns"].data_ptr(), down_global_ptr=tables["dng"].data_ptr(),
    gate_up_bias_ptr=0, down_bias_ptr=0, swiglu_alpha=1.702, swiglu_limit=float("inf"),
    core_ids=cores, task_timing_enabled=True)

io = []
for l in range(L):
    x = torch.randn(1, H).to(torch.bfloat16)
    ids = torch.full((1, K), -1, dtype=torch.int32)
    w = torch.full((1, K), 0.1, dtype=torch.float32)
    y = torch.empty(1, H, dtype=torch.bfloat16)
    io.append((x, ids, w, y, ex.create_task(l, 1, x.data_ptr(), ids.data_ptr(), w.data_ptr(), y.data_ptr())))

gen = torch.Generator().manual_seed(1)
def token():
    for x, ids, w, y, task in io:
        ids.fill_(-1)
        ids[0, :cold] = torch.randperm(E, generator=gen)[:cold].to(torch.int32)
        ex.run_task(task)

for _ in range(30): token()
ex.step_timing_snapshot_and_reset()
t0 = time.perf_counter()
for _ in range(tokens): token()
wall = (time.perf_counter() - t0) / tokens
snap = ex.step_timing_snapshot_and_reset()
agg = {k: sum(v[k] for v in snap.values()) for k in ("wake_us", "compute_us", "signal_us", "groups_us", "bytes", "tasks")}
out = dict(backing=backing, threads=n, cores=cores, cold=cold, tokens=tokens,
           token_ms=round(wall * 1e3, 3),
           wake_ms_per_token=round(agg["wake_us"] / tokens / 1e3, 3),
           compute_ms_per_token=round(agg["compute_us"] / tokens / 1e3, 3),
           signal_ms_per_token=round(agg["signal_us"] / tokens / 1e3, 3),
           mb_per_token=round(agg["bytes"] / tokens / 1e6, 1),
           compute_gbs=round(agg["bytes"] / (agg["compute_us"] * 1e3), 1),
           wall_gbs=round(agg["bytes"] / tokens / wall / 1e9, 1))
print(json.dumps(out), flush=True)
