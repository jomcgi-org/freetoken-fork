"""MLA latent-KV pools (GLM-5.2 / DeepSeek-style latent attention).

``MLAKVCache`` is the MHA pool's sibling for latent-KV models: ONE slab holding the
per-token latent ``ckv (kv_lora_rank) | kpe (qk_rope_head_dim)`` -- there is no
separate V (``v_cache`` aliases ``k_cache``, same convention as dsv4_paged_pool's
single-latent tiers). ``DSAKVCache`` extends it with DeepSeek-Sparse-Attention
index state. GLM-5.2 has one ``index_head_dim``-wide bf16 key row per token. A
pooled GLM-5.3 row is ``key | compression_gate`` and lives in the same backing
token record as all MLA latents. Both layouts use the same physical rows as the
latent cache, and rebuild resizes all state atomically.

Storage lives here, not in the attention backend, so the engine's rebuild path
(``MHAKVCache.rebuild``-shaped: fresh allocation, object identity preserved, views
re-derived by callers per forward) and the KV cost model (which budgets the index-K
bytes off the attention-group spec) stay correct by construction.
"""

from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """Paged latent-KV pool: ``[1, num_layers, num_pages, page_size, 1, latent_dim]``.

    The leading singleton keeps the buffer shape-compatible with MHAKVCache's
    (tokens = shape[2] * shape[3]).
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: tuple[int, ...] | None = None,
    ) -> None:
        self._latent_dim = latent_dim
        self._layer_ids = layer_ids or tuple(range(num_layers))
        self._num_layers = len(self._layer_ids)
        self._local_index = {layer_id: i for i, layer_id in enumerate(self._layer_ids)}
        self._page_size = page_size
        self._dtype = dtype
        self._device = device
        self._alloc(num_pages)

    def _alloc(self, num_pages: int) -> None:
        self._num_pages = num_pages
        self._kv_buffer = torch.empty(
            (1, self._num_layers, num_pages, self._page_size, 1, self._latent_dim),
            device=self._device,
            dtype=self._dtype,
        )

    # -- views ------------------------------------------------------------------
    def k_cache(self, layer_id: int) -> torch.Tensor:
        """Paged latent view ``[num_pages, page_size, latent_dim]``."""
        return self._kv_buffer[0, self._local_index[layer_id]].view(
            self._num_pages, self._page_size, -1
        )

    def v_cache(self, layer_id: int) -> torch.Tensor:
        # MLA: K == V (single latent); same buffer, dsv4_paged_pool precedent.
        return self.k_cache(layer_id)

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        """Row-flat latent view ``[num_pages * page_size, latent_dim]``."""
        return self._kv_buffer[0, self._local_index[layer_id]].view(
            -1, self._latent_dim
        )

    # -- writes -----------------------------------------------------------------
    def store_kv(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor | None,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter this forward's latent rows: ``c_kv`` [T, kv_lora_rank] and
        ``k_rope`` [T, qk_rope_head_dim] land in the row's two halves. ``k_rope``
        is None for a nope-only model, whose latent row is entirely ``c_kv``.

        v0: two narrow ``index_put_`` scatters. TODO: generalize kernel/csrc
        store.cu to a two-width fused store and route this through it.
        """
        rows = self.latent_rows(layer_id)
        if k_rope is None:
            assert rows.shape[1] == c_kv.shape[-1], (rows.shape, c_kv.shape)
            rows[out_loc] = c_kv
            return
        split = rows.shape[1] - k_rope.shape[-1]
        rows[out_loc, :split] = c_kv
        rows[out_loc, split:] = k_rope

    def rebuild(self, num_pages: int) -> None:
        """In-place resize (frees the old slab first; object identity preserved --
        callers re-derive views per forward, same contract as MHAKVCache.rebuild)."""
        self._kv_buffer = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._alloc(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(
            num_pages + 1
        )  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        return int(buf.numel() * buf.element_size()) // (
            self._num_pages * self._page_size
        ), 0

    # -- pool properties ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


class DSAKVCache(MLAKVCache):
    """MLA latent pool plus DSA index state.

    With ``index_kpool == 1`` this is the original independent bf16 key slab,
    unchanged. A pooled indexer caches ``cat(key, compression_gate)`` per token.
    Those rows share one backing record with the token-granular MLA latent so a
    radix page, disk prefix snapshot, and lazy page restore always move both state
    families together. Slot order is the backend's full-layer order.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        layer_ids: tuple[int, ...] | None = None,
        index_kpool: int = 1,
    ) -> None:
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_kpool = max(1, int(index_kpool))
        self._index_state_dim = index_head_dim * (2 if self._index_kpool > 1 else 1)
        if self._index_kpool > 1 and dtype.itemsize != 2:
            raise ValueError(
                f"pooled DSA index state is budgeted at 2 bytes per element, got {dtype}"
            )
        super().__init__(
            latent_dim, num_layers, num_pages, page_size, dtype, device, layer_ids
        )

    def _alloc(self, num_pages: int) -> None:
        if self._index_kpool > 1:
            # One physical token record contains every MLA layer's latent followed
            # by every full indexer's raw key and compression gate. The generic
            # disk-prefix path snapshots ``_kv_buffer`` by physical row, so this
            # layout also makes pooled DSA state part of eager and lazy restores.
            self._num_pages = num_pages
            self._record_dim = (
                self._num_layers * self._latent_dim
                + self._num_index_layers * self._index_state_dim
            )
            self._kv_buffer = torch.empty(
                (1, 1, num_pages, self._page_size, 1, self._record_dim),
                device=self._device,
                dtype=self._dtype,
            )
            for slot in range(self._num_index_layers):
                self._pooled_index_view(slot).zero_()
            return

        # The kpool-1 allocation is deliberately the original layout.
        super()._alloc(num_pages)
        # bf16 == the 2 bytes/token/layer the KV cost model budgets for this slab
        # (cache_status._kv_cost_model); keep the two in lockstep.
        self._index_k_buffer = torch.zeros(
            self._num_index_layers,
            num_pages * self._page_size,
            self._index_head_dim,
            dtype=torch.bfloat16,
            device=self._device,
        )

    def _pooled_index_view(self, slot: int) -> torch.Tensor:
        start = self._num_layers * self._latent_dim + slot * self._index_state_dim
        return self._kv_buffer[
            0, 0, :, :, 0, start : start + self._index_state_dim
        ].reshape(-1, self._index_state_dim)

    def k_cache(self, layer_id: int) -> torch.Tensor:
        if self._index_kpool == 1:
            return super().k_cache(layer_id)
        start = self._local_index[layer_id] * self._latent_dim
        return self._kv_buffer[0, 0, :, :, 0, start : start + self._latent_dim]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self.k_cache(layer_id)

    def latent_rows(self, layer_id: int) -> torch.Tensor:
        return self.k_cache(layer_id).reshape(-1, self._latent_dim)

    def rebuild(self, num_pages: int) -> None:
        if self._index_kpool == 1:
            self._index_k_buffer = None
        super().rebuild(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        per_page, fixed, page_tokens, reserve = super().kv_cost(config)
        args = getattr(config.model_config, "glm_dsa_args", None)
        if args is not None and getattr(args, "index_kpool", 1) > 1:
            # spec_kv_bytes_per_token already includes one bf16 key. Pooled GLM
            # also caches one bf16 gate vector of the same width per token.
            specs = [s for s in config.model_config.kv_cache_group_specs() if s.mla]
            extra = sum(s.index_head_dim * s.num_index_layers * 2 for s in specs)
            per_page += extra * config.page_size
        return per_page, fixed, page_tokens, reserve

    def unit_bytes(self) -> tuple[int, int]:
        if self._index_kpool > 1:
            # The joint record already includes latent, key, and gate state.
            return super().unit_bytes()
        # The index slab rides the same token budget as the latent slab; each slab's per-token
        # cost is floor-divided on its own, matching the cost model's two separate terms.
        kv, swa = super().unit_bytes()
        idx = self._index_k_buffer
        tokens = self._num_pages * self._page_size
        return kv + int(idx.numel() * idx.element_size()) // tokens, swa

    def index_k_cache(self, slot: int) -> torch.Tensor:
        """Row-flat token index state for a full-indexer layer slot.

        Width is ``index_head_dim`` for kpool 1 and ``2 * index_head_dim``
        (key followed by compression gate) for a pooled indexer.
        """
        if self._index_kpool > 1:
            return self._pooled_index_view(slot)
        return self._index_k_buffer[slot]

    def store_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, slot: int) -> None:
        self.index_k_cache(slot)[out_loc] = k

    @property
    def index_kpool(self) -> int:
        return self._index_kpool


__all__ = ["MLAKVCache", "DSAKVCache"]
