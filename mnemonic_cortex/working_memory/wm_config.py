"""
Plain-language summary
----------------------
What this file is for: Configuration objects for QDT working memory.
How it fits in the system: Knobs that size and enable WM features.
Status: ACTIVE
Important notes for non-coders: Change here to resize WM behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal


QDTProfileName = Literal["compact", "single_gpu_8_12gb", "deep"]


@dataclass(frozen=True)
class QDTWorkingMemoryCapacityEstimate:
    """Static footprint estimate for a QDT-WM configuration."""

    profile_name: str
    input_dim: int
    hidden_dim: int
    num_depths: int
    triplet_dim: int
    num_slots: int
    num_heads: int
    transformer_layers: int
    maae_transformer_layers: int
    cross_model_attention_layers: int
    context_tokens: int
    depth_state_scalars_per_token: int
    slot_backing_scalars: int
    qh_backing_scalars: int
    attention_context_scalars: int
    estimated_activation_scalars_per_batch_token: int
    estimated_fp16_mebibytes_per_batch_token: float
    estimated_fp32_mebibytes_per_batch_token: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    enable_inter_manifold_attention: bool = True
    inter_manifold_residual_mix: float = 0.15
    hardware_profile: str = "custom"
    qspin_guarded_shadow: bool = False
    qspin_source_matrix_complete: bool = True
    qspin_rollback_evidence_present: bool = True
    qspin_live_activation: bool = False
    qspin_live_mode: str = "disabled"
    qspin_live_kill_switch_enabled: bool = True
    qspin_live_allow_routing: bool = False
    qspin_live_allow_payload_transfer: bool = False
    qspin_live_allow_shared_slot_write: bool = False
    qspin_live_allow_qh_storage_write: bool = False
    qspin_live_allow_commit_execution: bool = False
    qspin_live_max_payload_tokens: int = 8
    qspin_live_payload_scale: float = 0.05
    qspin_live_routing_scale: float = 0.10
    eps: float = 1e-8

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    @classmethod
    def from_hardware_profile(
        cls,
        profile_name: QDTProfileName,
        *,
        input_dim: int | None = None,
    ) -> "QDTWorkingMemoryConfig":
        key = str(profile_name).strip().lower()
        if key == "compact":
            dim = int(input_dim or 160)
            return cls(
                input_dim=dim,
                hidden_dim=192,
                num_depths=8,
                num_slots=32,
                num_heads=cls._pick_num_heads(dim),
                transformer_layers=2,
                maae_transformer_layers=2,
                cross_model_attention_layers=3,
                context_tokens=48,
                hardware_profile="compact",
            )
        if key == "single_gpu_8_12gb":
            dim = int(input_dim or 256)
            return cls(
                input_dim=dim,
                hidden_dim=384,
                num_depths=8,
                num_slots=64,
                num_heads=cls._pick_num_heads(dim),
                transformer_layers=3,
                maae_transformer_layers=3,
                cross_model_attention_layers=4,
                context_tokens=64,
                hardware_profile="single_gpu_8_12gb",
                qspin_guarded_shadow=True,
                qspin_live_activation=True,
                qspin_live_mode="experimental_live",
                qspin_live_allow_routing=True,
                qspin_live_allow_payload_transfer=True,
                qspin_live_allow_shared_slot_write=True,
                qspin_live_allow_qh_storage_write=True,
                qspin_live_allow_commit_execution=True,
            )
        if key == "deep":
            dim = int(input_dim or 256)
            return cls(
                input_dim=dim,
                hidden_dim=512,
                num_depths=8,
                num_slots=96,
                num_heads=cls._pick_num_heads(dim),
                transformer_layers=4,
                maae_transformer_layers=4,
                cross_model_attention_layers=5,
                context_tokens=96,
                hardware_profile="deep",
                qspin_guarded_shadow=True,
                qspin_live_activation=False,
                qspin_live_mode="disabled",
            )
        raise ValueError(f"unknown QDT hardware profile: {profile_name}")

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
        if self.context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        self.qspin_live_max_payload_tokens = int(max(1, self.qspin_live_max_payload_tokens))
        self.qspin_live_payload_scale = float(min(1.0, max(0.0, self.qspin_live_payload_scale)))
        self.qspin_live_routing_scale = float(min(1.0, max(0.0, self.qspin_live_routing_scale)))
        mode = str(self.qspin_live_mode).strip().lower()
        if self.qspin_live_activation and mode != "experimental_live":
            raise ValueError("qspin_live_activation requires qspin_live_mode='experimental_live'")
        if not self.qspin_live_activation and mode not in {"disabled", "experimental_live"}:
            raise ValueError("qspin_live_mode must be disabled or experimental_live")
        self.qspin_live_mode = mode
        if not 0.0 <= self.residual_fusion_weight <= 1.0:
            raise ValueError("residual_fusion_weight must be in [0,1]")
        if not 0.0 <= float(self.inter_manifold_residual_mix) <= 1.0:
            raise ValueError("inter_manifold_residual_mix must be in [0,1]")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def capacity_estimate(self, *, batch_size: int = 1, seq_len: int = 1) -> QDTWorkingMemoryCapacityEstimate:
        self.validate()
        b = max(1, int(batch_size))
        t = max(1, int(seq_len))
        depth_state_scalars = int(self.num_depths * self.triplet_dim * self.input_dim)
        slot_backing_scalars = int(self.num_slots * self.input_dim)
        qh_backing_scalars = int(self.num_slots * self.num_depths * self.input_dim)
        attention_context_scalars = int(max(0, self.context_tokens) * self.input_dim)
        per_batch_token = depth_state_scalars * max(1, self.transformer_layers * 2)
        total_activation_scalars = b * t * per_batch_token
        total_static_scalars = slot_backing_scalars + qh_backing_scalars + attention_context_scalars
        total_scalars = total_activation_scalars + total_static_scalars
        return QDTWorkingMemoryCapacityEstimate(
            profile_name=str(self.hardware_profile),
            input_dim=int(self.input_dim),
            hidden_dim=int(self.hidden_dim),
            num_depths=int(self.num_depths),
            triplet_dim=int(self.triplet_dim),
            num_slots=int(self.num_slots),
            num_heads=int(self.num_heads),
            transformer_layers=int(self.transformer_layers),
            maae_transformer_layers=int(self.maae_transformer_layers),
            cross_model_attention_layers=int(self.cross_model_attention_layers),
            context_tokens=int(self.context_tokens),
            depth_state_scalars_per_token=depth_state_scalars,
            slot_backing_scalars=slot_backing_scalars,
            qh_backing_scalars=qh_backing_scalars,
            attention_context_scalars=attention_context_scalars,
            estimated_activation_scalars_per_batch_token=per_batch_token,
            estimated_fp16_mebibytes_per_batch_token=float(total_scalars * 2) / (1024.0 * 1024.0),
            estimated_fp32_mebibytes_per_batch_token=float(total_scalars * 4) / (1024.0 * 1024.0),
        )


def qdt_config_from_hardware_profile(profile_name: QDTProfileName, *, input_dim: int | None = None) -> QDTWorkingMemoryConfig:
    return QDTWorkingMemoryConfig.from_hardware_profile(profile_name, input_dim=input_dim)
