from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    k = k.view(k.shape[0], -1)
    v = v.view(v.shape[0], -1)
    if k_cache.dtype != k.dtype or v_cache.dtype != v.dtype:
        k = k.to(k_cache.dtype)
        v = v.to(v_cache.dtype)

    # PyTorch index_copy_ requires int64 indices and has no float8 kernel. The
    # inputs are converted above, then the JIT store kernel performs a pure
    # row-byte copy. It accepts int32 or int64 indices and keys its
    # specialization on this original byte count. Passing uint8 views keeps TVM
    # FFI independent of float8 dtype support as well.
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        indices,
        k.view(torch.uint8),
        v.view(torch.uint8),
    )
