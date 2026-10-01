"""Bit-exactness of CPU MoE decode across worker counts (bs 1 and 2). Prints a digest per thread count."""
import hashlib, sys, torch
sys.argv = ["x", "thp", "1", "4", "1"]
src = open(__file__.replace("thread-parity.py", "cpumoe-micro.py")).read().split("n, cores = resolve")[0]
exec(src)  # builds banks/tables (MICRO_E experts, random, e4m3-sane scales)
for threads in (1, 2, 8, 14, 16):
    n, cores = resolve_threads_and_affinity(threads)
    ex = _cpu_moe.CpuMoeExecutor(num_threads=n, num_layers=L, num_experts=E, top_k=K, hidden_size=H, inter_size=I,
        max_tokens=4, activation_id=_ACT_IDS["silu"], apply_router_weight_on_input=0, weight_format=_WFMT_IDS["nvfp4"],
        gate_up_ptr=tables["gup"].data_ptr(), down_ptr=tables["dnp"].data_ptr(),
        gate_up_scale_ptr=tables["gus"].data_ptr(), gate_up_global_ptr=tables["gug"].data_ptr(),
        down_scale_ptr=tables["dns"].data_ptr(), down_global_ptr=tables["dng"].data_ptr(),
        gate_up_bias_ptr=0, down_bias_ptr=0, swiglu_alpha=1.702, swiglu_limit=float("inf"), core_ids=cores)
    h = hashlib.sha256()
    g = torch.Generator().manual_seed(7)
    for bs in (1, 2):
        for l in range(L):
            x = (torch.randn(bs, H, generator=g) * 0.5).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(bs)]).to(torch.int32)
            ids[:, 6:] = -1
            w = torch.rand(bs, K, generator=g)
            y = torch.empty(bs, H, dtype=torch.bfloat16)
            t = ex.create_task(l, bs, x.data_ptr(), ids.data_ptr(), w.data_ptr(), y.data_ptr())
            ex.run_task(t)
            assert torch.isfinite(y.float()).all()
            h.update(y.view(torch.int16).numpy().tobytes())
    print(threads, h.hexdigest()[:16], flush=True)
    del ex
