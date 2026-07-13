"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: wm depth adapter.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import hashlib

import torch

from .depth_indexed_slot_lattice import DepthIndexedSlotLattice
from .depth_lattice_config import DepthLatticeConfig
from .depth_lattice_types import DepthReadMode, DepthRole, DepthWriteMode, DepthWriteProposal, QHDepthCode
from .depth_trace import DepthLatticeTrace


class WMDepthAdapterError(ValueError):
    """Raised when the WM depth adapter receives invalid input."""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WMDepthAdapterConfig:
    """Optional working-memory depth adapter configuration.

    The adapter is inert unless enabled=True. Even when enabled, writes are
    proposal-only by default.
    """

    enabled: bool = False
    input_dim: int = 256
    value_dim: int = 256
    slot_count: int = 64
    read_top_k_slots: int = 8
    read_top_k_depths: int = 2
    context_binding_depth: int = 3
    temporal_episode_depth: int = 5
    volatile_trace_depth: int = 7
    no_mutation_by_default: bool = True
    finite_checks: bool = True

    @classmethod
    def disabled(cls, input_dim: int = 256, value_dim: Optional[int] = None) -> "WMDepthAdapterConfig":
        return cls(enabled=False, input_dim=input_dim, value_dim=value_dim or input_dim)

    @classmethod
    def enabled_default(cls, input_dim: int = 256, value_dim: Optional[int] = None, slot_count: int = 64) -> "WMDepthAdapterConfig":
        return cls(enabled=True, input_dim=input_dim, value_dim=value_dim or input_dim, slot_count=slot_count)

    def to_lattice_config(self) -> DepthLatticeConfig:
        return DepthLatticeConfig.wm(
            slot_count=self.slot_count,
            key_dim=self.input_dim,
            value_dim=self.value_dim,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "input_dim": self.input_dim,
            "value_dim": self.value_dim,
            "slot_count": self.slot_count,
            "read_top_k_slots": self.read_top_k_slots,
            "read_top_k_depths": self.read_top_k_depths,
            "context_binding_depth": self.context_binding_depth,
            "temporal_episode_depth": self.temporal_episode_depth,
            "volatile_trace_depth": self.volatile_trace_depth,
            "no_mutation_by_default": self.no_mutation_by_default,
            "finite_checks": self.finite_checks,
        }


@dataclass
class WMDepthAdapter:
    """Optional adapter connecting WM inputs to a DepthIndexedSlotLattice.

    This adapter is additive. It does not replace QDTWorkingMemory and it does
    not mutate permanent memory by default.
    """

    config: WMDepthAdapterConfig
    lattice: Optional[DepthIndexedSlotLattice] = None

    def __post_init__(self) -> None:
        if self.config.input_dim <= 0 or self.config.value_dim <= 0:
            raise WMDepthAdapterError("input_dim/value_dim must be positive")
        if self.config.slot_count <= 0:
            raise WMDepthAdapterError("slot_count must be positive")
        for name in ("context_binding_depth", "temporal_episode_depth", "volatile_trace_depth"):
            value = getattr(self.config, name)
            if not (0 <= int(value) < 8):
                raise WMDepthAdapterError(f"{name} must be in [0,7]")
        if self.lattice is None:
            lattice_cfg = self.config.to_lattice_config()
            object.__setattr__(lattice_cfg, "read_top_k_slots", self.config.read_top_k_slots) if False else None
            # Dataclass is frozen, so create explicit config with read parameters.
            lattice_cfg = DepthLatticeConfig(
                bank_id="wm.depth_lattice",
                bank_kind=lattice_cfg.bank_kind,
                slot_count=self.config.slot_count,
                num_depths=8,
                key_dim=self.config.input_dim,
                value_dim=self.config.value_dim,
                read_top_k_slots=min(self.config.read_top_k_slots, self.config.slot_count),
                read_top_k_depths=self.config.read_top_k_depths,
                finite_checks=self.config.finite_checks,
                no_mutation_by_default=self.config.no_mutation_by_default,
            )
            self.lattice = DepthIndexedSlotLattice(lattice_cfg)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def process(self, wm_state: torch.Tensor, *, return_trace: bool = False):
        """Process WM state through optional depth read.

        Disabled mode returns the input unchanged with a pass-through trace.
        Enabled mode performs a depth-lattice read and returns a depth summary.
        """

        self._validate_wm_state(wm_state)
        if not self.enabled:
            trace = self._pass_through_trace(wm_state)
            if return_trace:
                return wm_state, trace
            return wm_state

        read_value, attention, trace = self.lattice.read(wm_state, read_mode=DepthReadMode.TOP_K, return_trace=True)
        if return_trace:
            return read_value, trace
        return read_value

    def propose_context_candidate_writes(
        self,
        *,
        context: torch.Tensor,
        response: Optional[torch.Tensor] = None,
        candidate: Optional[Any] = None,
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route compressed context/response content into depth write proposals.

        Routing:
        - Z3 contextual binding
        - Z5 temporal episode
        - Z7 volatile trace

        Returns proposals only. It does not commit writes.
        """

        self._validate_wm_state(context)
        if response is not None:
            self._validate_wm_state(response)

        context_vec = context.mean(dim=1) if context.dim() == 3 else context
        response_vec = response.mean(dim=1) if isinstance(response, torch.Tensor) and response.dim() == 3 else response
        value = context_vec[0] if context_vec.dim() == 2 else context_vec.reshape(-1, self.config.value_dim)[0]
        if value.numel() != self.config.value_dim:
            raise WMDepthAdapterError("context value dim must match adapter value_dim")

        key = value[: self.config.input_dim]
        if key.numel() != self.config.input_dim:
            raise WMDepthAdapterError("context key dim must match adapter input_dim")

        content_hash = self._candidate_hash(candidate, project_id, chat_id, episode_id)
        canonical_slot_id = f"wm.context.{content_hash[:16]}"
        slot_index = int(int(content_hash[:8], 16) % self.config.slot_count)

        qh_code = QHDepthCode(
            depth_code="Z3/Z5/Z7",
            bank_code="WM_DEPTH",
            geometry_code="contextual_curved",
            triplet_code="anchor_direction_phase",
            memory_type_code="context_candidate",
            task_mode_code="working_memory_reasoning",
        )

        proposals = {
            "contextual_binding": self.lattice.propose_write(
                slot_index=slot_index,
                depth_index=self.config.context_binding_depth,
                value=value,
                key=key,
                mode=DepthWriteMode.SINGLE,
                canonical_slot_id=canonical_slot_id,
                qh_code=qh_code,
            ).to_dict(),
            "temporal_episode": self.lattice.propose_write(
                slot_index=slot_index,
                depth_index=self.config.temporal_episode_depth,
                value=value,
                key=key,
                mode=DepthWriteMode.SINGLE,
                canonical_slot_id=canonical_slot_id,
                qh_code=qh_code,
            ).to_dict(),
            "volatile_trace": self.lattice.propose_write(
                slot_index=slot_index,
                depth_index=self.config.volatile_trace_depth,
                value=value,
                key=key,
                mode=DepthWriteMode.SINGLE,
                canonical_slot_id=canonical_slot_id,
                qh_code=qh_code,
            ).to_dict(),
        }
        return {
            "committed": False,
            "shadow_only": True,
            "canonical_slot_id": canonical_slot_id,
            "slot_index": slot_index,
            "depth_routes": {
                "contextual_binding": self.config.context_binding_depth,
                "temporal_episode": self.config.temporal_episode_depth,
                "volatile_trace": self.config.volatile_trace_depth,
            },
            "proposals": proposals,
            "metadata": {
                "project_id": project_id,
                "chat_id": chat_id,
                "episode_id": episode_id,
                "candidate_present": candidate is not None,
                "response_present": response is not None,
                "no_permanent_mutation_without_permission": True,
            },
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": True,
                "write_permission_granted": False,
                "no_memory_store_mutation": True,
                "wm_depth_candidate_routing": True,
            },
        }

    def _validate_wm_state(self, wm_state: torch.Tensor) -> None:
        if not isinstance(wm_state, torch.Tensor):
            raise WMDepthAdapterError("wm_state must be a torch.Tensor")
        if wm_state.dim() not in {2, 3}:
            raise WMDepthAdapterError("wm_state must be [B,D] or [B,T,D]")
        if wm_state.size(-1) != self.config.input_dim:
            raise WMDepthAdapterError(f"wm_state last dim must be {self.config.input_dim}")
        if self.config.finite_checks and not torch.isfinite(wm_state).all():
            raise WMDepthAdapterError("wm_state contains NaN/Inf")

    def _pass_through_trace(self, wm_state: torch.Tensor) -> Dict[str, Any]:
        return {
            "trace_type": "wm_depth_adapter_trace",
            "enabled": False,
            "pass_through": True,
            "input_shape": list(wm_state.shape),
            "output_shape": list(wm_state.shape),
            "config": self.config.to_dict(),
            "paamax_metadata": {
                "trace_governance": True,
                "write_permission_required": False,
                "no_memory_store_mutation": True,
                "wm_depth_adapter": True,
            },
        }

    def _candidate_hash(self, candidate: Optional[Any], project_id: Optional[str], chat_id: Optional[str], episode_id: Optional[str]) -> str:
        payload = {
            "candidate": repr(candidate)[:2048],
            "project_id": project_id,
            "chat_id": chat_id,
            "episode_id": episode_id,
        }
        return _hash_text(repr(payload))


def wm_depth_contract() -> Dict[str, Any]:
    return {
        "module": "wm_depth_adapter",
        "stage": "REASON-1B",
        "optional": True,
        "default_enabled": False,
        "input_shapes": ["[B,D]", "[B,T,D]"],
        "output_shapes": ["pass-through [B,D]/[B,T,D] when disabled", "depth summary [B,V] when enabled"],
        "write_behavior": "shadow proposals only by default",
        "depth_routes": {
            "contextual_binding": 3,
            "temporal_episode": 5,
            "volatile_trace": 7,
        },
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
