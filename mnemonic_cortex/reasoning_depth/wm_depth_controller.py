"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: wm depth controller.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .depth_capacity_metrics import compute_depth_capacity_metrics
from .wm_depth_adapter import WMDepthAdapter, WMDepthAdapterConfig, WMDepthAdapterError


class WMDepthControllerError(ValueError):
    """Raised when the WM depth controller cannot route safely."""


@dataclass
class WMDepthController:
    """Small controller for optional working-memory depth routing.

    The controller owns the adapter and exposes a stable interface for later
    QDTWorkingMemory integration. Default behavior is disabled/pass-through.
    """

    adapter: WMDepthAdapter = field(default_factory=lambda: WMDepthAdapter(WMDepthAdapterConfig.disabled()))

    @classmethod
    def disabled(cls, input_dim: int = 256, value_dim: Optional[int] = None) -> "WMDepthController":
        return cls(adapter=WMDepthAdapter(WMDepthAdapterConfig.disabled(input_dim=input_dim, value_dim=value_dim)))

    @classmethod
    def enabled_default(cls, input_dim: int = 256, value_dim: Optional[int] = None, slot_count: int = 64) -> "WMDepthController":
        return cls(adapter=WMDepthAdapter(WMDepthAdapterConfig.enabled_default(input_dim=input_dim, value_dim=value_dim, slot_count=slot_count)))

    @property
    def enabled(self) -> bool:
        return self.adapter.enabled

    def process_wm(self, wm_state: torch.Tensor, *, return_trace: bool = False):
        return self.adapter.process(wm_state, return_trace=return_trace)

    def route_context_candidate(
        self,
        *,
        context: torch.Tensor,
        response: Optional[torch.Tensor] = None,
        candidate: Optional[Any] = None,
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.adapter.propose_context_candidate_writes(
            context=context,
            response=response,
            candidate=candidate,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
        )

    def capacity_metrics(self) -> Dict[str, Any]:
        return self.adapter.lattice.capacity_metrics()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "adapter_config": self.adapter.config.to_dict(),
            "capacity_metrics": self.capacity_metrics(),
            "safety": {
                "destructive_qdt_replacement": False,
                "automatic_memory_store_mutation": False,
                "shadow_writes_only_by_default": True,
            },
        }


def attach_wm_depth_controller(target: Any, controller: Optional[WMDepthController] = None, *, attr_name: str = "wm_depth_controller") -> Any:
    """Attach a disabled-by-default WMDepthController to a shell object.

    This avoids patching real QDTWorkingMemory destructively. Later integration
    stages can use this helper when a concrete cortex/WM shell is available.
    """

    if target is None:
        raise WMDepthControllerError("target cannot be None")
    if controller is None:
        controller = WMDepthController.disabled()
    if hasattr(target, attr_name):
        existing = getattr(target, attr_name)
        if existing is not None:
            return target
    setattr(target, attr_name, controller)
    return target


def wm_depth_controller_contract() -> Dict[str, Any]:
    return {
        "module": "wm_depth_controller",
        "stage": "REASON-1B",
        "default_enabled": False,
        "integration_mode": "attach helper / optional controller, no destructive QDT replacement",
        "context_candidate_depth_routes": {
            "Z3": "contextual_binding",
            "Z5": "temporal_episode",
            "Z7": "volatile_trace",
        },
        "paamax_metadata": {
            "trace_governance": True,
            "write_permission_required": True,
            "write_permission_granted": False,
        },
    }
