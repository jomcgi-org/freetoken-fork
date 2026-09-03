"""CLI: convert an HF safetensors checkpoint to a FreeToken Weight (FTW) checkpoint.

    ft checkpoint --model <hf_dir> --out <ftw_dir> \
        [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8]
        [--speculative-mtp on --mtp-quant {bf16,nvfp4}] [--gpu <uuid-or-index>]

The output dir is self-contained: point the server's ``--model`` at it to load via the FTW
fast path (auto-detected).
"""

from __future__ import annotations

import argparse
import time

import torch

from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, single_gpu_arg

from .convert import convert_checkpoint

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main(argv: list[str] | None = None, prog: str = "freetoken.checkpoint") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("--model", required=True, help="source HF safetensors checkpoint dir")
    p.add_argument("--out", required=True, help="output FTW checkpoint dir")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--moe-backend", default="offload",
                   help="offload (experts -> banks) or e.g. triton (experts stay dense)")
    p.add_argument("--shard-gib", type=float, default=8.0, help="max shard size in GiB")
    p.add_argument(
        "--speculative-mtp",
        choices=("off", "on"),
        default="off",
        help="preserve the optional Qwen3.8 MTP head in the FTW (default: off)",
    )
    p.add_argument(
        "--mtp-quant",
        choices=("bf16", "nvfp4"),
        default="bf16",
        help="resident MTP routed-expert precision (default: bf16)",
    )
    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="optional GPU for the hardware-specific NVFP4 repack: a GPU UUID "
                        "(GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index "
                        "(default: CPU, preserving the native NVFP4 layout)")
    p.add_argument(
        "--moe-activation-dtype",
        choices=("auto", "bf16", "nvfp4"),
        default="auto",
        help=(
            "SM120 expert activation layout for a GPU repack. auto uses NVFP4 when "
            "all ModelOpt input scales are present; explicit nvfp4 fails if unsupported."
        ),
    )
    ns = p.parse_args(argv)
    if ns.speculative_mtp == "off" and ns.mtp_quant != "bf16":
        p.error("--mtp-quant nvfp4 requires --speculative-mtp on")

    device = "cpu"
    if ns.gpu is not None:
        # Same as ft serve --gpu: resolve, then bind by UUID at CUDA init. With no
        # --gpu the converter remains CPU-only and writes checkpoint-native NVFP4 banks.
        try:
            assign_gpu(ns.gpu)
            device = f"cuda:{bind_assigned_gpu().index}"
        except (ValueError, RuntimeError) as e:
            p.error(str(e))

    shard_limit = int(ns.shard_gib * (1 << 30))
    shard_limit -= shard_limit % 4096  # keep aligned
    t = time.perf_counter()
    index = convert_checkpoint(
        ns.model, ns.out, dtype=_DTYPES[ns.dtype],
        moe_backend=ns.moe_backend, shard_limit=shard_limit, device=device,
        include_mtp=ns.speculative_mtp == "on",
        mtp_quant=ns.mtp_quant,
        moe_activation_dtype=ns.moe_activation_dtype,
    )
    dt = time.perf_counter() - t
    c = index["counts"]
    gib = index["total_bytes"] / (1 << 30)
    print(f"\nwrote FTW checkpoint -> {ns.out}")
    print(
        f"  tensors: {c['weight']} weight + {c.get('mtp', 0)} mtp + "
        f"{c['experts_bank']} experts_bank"
    )
    print(f"  FTW: {gib:.2f} GiB across {len(index['shards'])} shard(s) "
          f"(<= {ns.shard_gib} GiB each)")
    print(f"  quant_format: {index['quant_format']}  fingerprint={index['fingerprint']}")
    if index.get("mtp_quant") is not None:
        print(f"  mtp_quant: {index['mtp_quant']}")
    print(f"  converted in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
