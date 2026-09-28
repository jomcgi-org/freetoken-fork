from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    # Layer-major prefill: run up to this many tokens of consecutive chunks of one
    # request layer by layer, so each layer's experts reach the GPU once per group
    # instead of once per chunk. The group's hyper-connection residual stays on the
    # GPU, so size it to free GPU memory. 0 disables it.
    prefill_layer_major_tokens: int = 0
    # Chunk size for prompts that need more than one chunk while layer-major prefill is
    # on. Within a group chunk size no longer sets weight traffic, and smaller chunks
    # leave GPU memory for a longer group. 0 keeps max_extend_tokens.
    prefill_layer_major_chunk: int = 0
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False
    # Waiting requests gain one effective-priority point per interval. 0 disables aging.
    priority_aging_seconds: float = 30.0

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
