"""Diagnostic: dump decode router inputs and logits per MoE layer.

``FREETOKEN_ROUTE_DUMP=<dir>`` records, for every eager single-token decode step, each
MoE layer's router input and router logits, and writes them in shards of
``FREETOKEN_ROUTE_DUMP_STEPS`` steps (default 256) to ``<dir>/routes-<pid>-<n>.pt``,
together with the router weights (first shard only), the layer residency and the HOT
expert mapping at flush time. Used offline to measure how well one layer's hidden
state predicts the next layer's cold experts. Decode must run eagerly
(``--cuda-graph-max-bs 0``): graph replays bypass the hook. Every record synchronizes.
"""

from __future__ import annotations

import os

import torch

_DIR = os.environ.get("FREETOKEN_ROUTE_DUMP", "")
_STEPS = int(os.environ.get("FREETOKEN_ROUTE_DUMP_STEPS", "256") or 256)


class _Recorder:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.rows: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.gates: dict[int, torch.Tensor] = {}
        self.shard = 0
        self.last_layer = -1
        self.steps = 0
        self.cache = None

    def record(self, layer_id: int, moe, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        if layer_id <= self.last_layer:
            self.steps += 1
            if self.steps >= _STEPS:
                self.flush()
        self.last_layer = layer_id
        if layer_id not in self.gates:
            self.gates[layer_id] = moe.gate.weight.detach().to("cpu", torch.float32)
        if self.cache is None:
            self.cache = getattr(moe.experts, "offload_cache", None)
        self.rows.setdefault(layer_id, []).append(
            (hidden[0].detach().to("cpu", torch.float32), logits[0].detach().to("cpu", torch.float32))
        )

    def flush(self) -> None:
        if not self.rows:
            return
        cache = self.cache
        payload = {
            "steps": self.steps,
            "hidden": {k: torch.stack([h for h, _ in v]) for k, v in self.rows.items()},
            "logits": {k: torch.stack([g for _, g in v]) for k, v in self.rows.items()},
            "residency": list(getattr(cache, "layer_residency", ()) or ()),
            "hot_mapping": (
                cache._hot_mapping_host.clone()
                if cache is not None and getattr(cache, "_hot_mapping_host", None) is not None
                else None
            ),
        }
        if self.shard == 0:
            payload["gates"] = self.gates
        path = os.path.join(self.directory, f"routes-{os.getpid()}-{self.shard}.pt")
        torch.save(payload, path)
        self.shard += 1
        self.rows = {}
        self.steps = 0


_RECORDER = _Recorder(_DIR) if _DIR else None


def maybe_record(layer_id, moe, hidden: torch.Tensor, logits: torch.Tensor) -> None:
    """Record one decode layer's router input and logits when the dump is on."""
    if _RECORDER is None or layer_id is None or hidden.shape[0] != 1:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    from freetoken.core import get_global_ctx

    batch = getattr(get_global_ctx(), "batch", None)
    if batch is None or not getattr(batch, "is_decode", False):
        return
    _RECORDER.record(int(layer_id), moe, hidden, logits)
