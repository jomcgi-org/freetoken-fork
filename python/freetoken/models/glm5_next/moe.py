from __future__ import annotations

from freetoken.models.glm_moe import derive_glm_moe_geometry, glm_shared_expert_count
from freetoken.models.glm_moe_dsa.moe import GlmMoeDsaSparseBlock

from .mlp import Glm5NextGatedMLP


class Glm5NextSparseBlock(GlmMoeDsaSparseBlock):
    """GLM sigmoid/noaux router plus clamped routed and resident shared SwiGLU."""

    def __init__(self, config, layer_id: int):
        super().__init__(config, layer_id)
        self.experts.activation = "clamped_silu"
        geometry = derive_glm_moe_geometry(config)
        self.shared_experts = Glm5NextGatedMLP(
            geometry.hidden_size,
            geometry.intermediate_size * glm_shared_expert_count(config),
            config.swiglu_limit,
        )


__all__ = ["Glm5NextSparseBlock"]
