from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class QDTWorkingMemoryConfig:
    """Top-level QDT-WM-MAAE working-memory configuration.

    This config is intentionally compact but complete enough for the assembled
    WM-2C stack:
    - curved core
    - quaternion depth replication
    - intra-depth transformer
    - cross-depth transformer
    - depth-specific addressing
    - depth fusion
    - triplet state
    - trace emission
    """

    input_dim: int
    hidden_dim: int = 128
    num_depths: int = 8
    triplet_dim: int = 3
    num_slots: int = 8
    num_heads: int = 0
    transformer_layers: int = 2
    maae_transformer_layers: int = 2
    cross_model_attention_layers: int = 4
    context_tokens: int = 0
    use_shadow_writes: bool = True
    residual_fusion_weight: float = 0.50
    eps: float = 1e-8

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    def validate(self) -> None:
        if self.num_heads <= 0:
            self.num_heads = self._pick_num_heads(self.input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_depths <= 0:
            raise ValueError("num_depths must be positive")
        if self.triplet_dim != 3:
            raise ValueError("triplet_dim must remain 3")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.input_dim % self.num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads for WM-2C transformers")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if self.maae_transformer_layers <= 0:
            raise ValueError("maae_transformer_layers must be positive")
        if self.cross_model_attention_layers <= 0:
            raise ValueError("cross_model_attention_layers must be positive")
        if not 0.0 <= self.residual_fusion_weight <= 1.0:
            raise ValueError("residual_fusion_weight must be in [0,1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
