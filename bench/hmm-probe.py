"""Probe: can a CUDA kernel (triton gather) read a file-backed mmap directly (HMM)?

Success => the GPU faults file pages in itself and rows match the CPU reference.
Failure modes: CUDA_ERROR_ILLEGAL_ADDRESS / hang / garbage rows.
Run on the VM: .venv python with the GPU free.
"""
import ctypes
import mmap
import os
import sys

import torch

from freetoken.kernel.triton.ple import ple_gather_rows_from_ptr

ROWS, DIM = 200_000, 160  # ~30 MiB of fp8 rows
PATH = "/mnt/nvme/hmm-probe.bin"

payload = (torch.arange(ROWS * DIM, dtype=torch.int64) % 120).to(torch.uint8)
with open(PATH, "wb") as f:
    f.write(payload.numpy().tobytes())

f = open(PATH, "rb")
m = mmap.mmap(f.fileno(), ROWS * DIM, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The given buffer is not writable")
    mm_tensor = torch.frombuffer(m, dtype=torch.uint8, count=ROWS * DIM)
addr = mm_tensor.data_ptr()
print(f"mmap at 0x{addr:x}")

ids = torch.randint(0, ROWS, (64,), dtype=torch.int32)
ids_pin = ids.pin_memory()
dst = torch.empty(64, DIM, dtype=torch.bfloat16, device="cuda")

try:
    ple_gather_rows_from_ptr(addr, ROWS, DIM, ids_pin.data_ptr(), 64, dst, 1.0, True)
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    print(f"HMM-PROBE: FAILED with {type(exc).__name__}: {exc}")
    sys.exit(1)

ref = (
    payload.view(ROWS, DIM)
    .index_select(0, ids.long())
    .view(torch.float8_e4m3fn)
    .to(torch.bfloat16)
)
ok = torch.equal(dst.cpu(), ref)
print(f"HMM-PROBE: {'SUCCESS - GPU read file-backed mmap correctly' if ok else 'MISMATCH - ran but data wrong'}")
os.unlink(PATH)
sys.exit(0 if ok else 2)
