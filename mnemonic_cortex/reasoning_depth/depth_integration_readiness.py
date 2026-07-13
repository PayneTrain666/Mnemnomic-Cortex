"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth integration readiness.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .depth_capacity_validation import DepthCapacityValidationConfig, validate_wm_mann_ltm_capacity
from .shared_depth_slot_registry import SharedDepthSlotRegistry
from .wm_depth_controller import WMDepthController
from .mann_depth_adapter import MANNDepthAdapter, MANNDepthAdapterConfig
from .ltm_depth_adapter import LTMDepthAdapter, LTMDepthAdapterConfig


@dataclass
class DepthIntegrationReadinessReport:
    """Readiness status for WM/MANN/LTM depth-lattice integration."""

    wm_ready: bool
    mann_ready: bool
    ltm_ready: bool
    registry_ready: bool
    default_inert_preserved: bool
    mutation_gates_preserved: bool
    remaining_deferred_work: List[str] = field(default_factory=list)
    readiness_level: str = "not_ready"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wm_ready": self.wm_ready,
            "mann_ready": self.mann_ready,
            "ltm_ready": self.ltm_ready,
            "registry_ready": self.registry_ready,
            "default_inert_preserved": self.default_inert_preserved,
            "mutation_gates_preserved": self.mutation_gates_preserved,
            "remaining_deferred_work": list(self.remaining_deferred_work),
            "readiness_level": self.readiness_level,
            "details": self.details,
            "safety": {
                "production_complete_claim": False,
                "destructive_replacement": False,
                "permanent_memory_store_mutation": False,
                "fake_quantum_hardware_claim": False,
            },
        }


def evaluate_depth_integration_readiness(config: Optional[DepthCapacityValidationConfig] = None) -> DepthIntegrationReadinessReport:
    cfg = config or DepthCapacityValidationConfig()
    cfg.validate()

    capacity = validate_wm_mann_ltm_capacity(cfg)

    wm = WMDepthController.disabled(input_dim=cfg.key_dim)
    mann = MANNDepthAdapter(MANNDepthAdapterConfig.disabled(key_dim=cfg.key_dim, value_dim=cfg.value_dim))
    ltm = LTMDepthAdapter(LTMDepthAdapterConfig.disabled(key_dim=cfg.key_dim, value_dim=cfg.value_dim))
    registry = SharedDepthSlotRegistry()
    record = registry.create_or_update(
        canonical_slot_id="readiness.registry.1",
        content="readiness",
        wm_ref="wm.readiness.1",
        mann_ref="mann.readiness.1",
        ltm_ref="ltm.readiness.1",
        source_stage="REASON-1E",
        source_pack="readiness_smoke",
        depth_roles_present=[1, 5],
    )

    wm_ready = bool(capacity["wm_available"])
    mann_ready = bool(capacity["mann_available"])
    ltm_ready = bool(capacity["ltm_available"])
    registry_payload = registry.to_dict()
    registry_ready = registry_payload["record_count"] == 1 and registry_payload["safety"]["shared_physical_tensor"] is False
    default_inert_preserved = (
        getattr(wm, "enabled", False) is False
        and mann.enabled is False
        and ltm.enabled is False
    )
    mutation_gates_preserved = (
        registry_payload["safety"]["shared_physical_tensor"] is False
        and capacity["safety_flags"]["permanent_memory_store_mutation"] is False
    )

    remaining = [
        "REASON-2A reasoning controller/orchestrator integration",
        "persistent consolidation commit gate implementation",
        "production-scale benchmark suite",
        "real LTM/MANN/SPCP source integration beyond additive adapters",
    ]
    ready_core = all([wm_ready, mann_ready, ltm_ready, registry_ready, default_inert_preserved, mutation_gates_preserved])
    readiness_level = "integration_ready_for_controller_design" if ready_core else "not_ready"

    return DepthIntegrationReadinessReport(
        wm_ready=wm_ready,
        mann_ready=mann_ready,
        ltm_ready=ltm_ready,
        registry_ready=registry_ready,
        default_inert_preserved=default_inert_preserved,
        mutation_gates_preserved=mutation_gates_preserved,
        remaining_deferred_work=remaining,
        readiness_level=readiness_level,
        details={
            "capacity": capacity,
            "registry_record": record.to_dict(),
            "default_states": {
                "wm_enabled": getattr(wm, "enabled", False),
                "mann_enabled": mann.enabled,
                "ltm_enabled": ltm.enabled,
            },
        },
    )


def depth_integration_readiness_contract() -> Dict[str, Any]:
    return {
        "module": "depth_integration_readiness",
        "stage": "REASON-1E",
        "readiness_fields": [
            "wm_ready",
            "mann_ready",
            "ltm_ready",
            "registry_ready",
            "default_inert_preserved",
            "mutation_gates_preserved",
            "remaining_deferred_work",
            "readiness_level",
        ],
        "production_complete_claim": False,
        "bounded": True,
        "no_mutation_by_default": True,
    }
