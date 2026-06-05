"""QSPIN production plan metadata contract (non-executable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class QSpinProductionPlan:
    plan_id: str = "qspin_prod_plan_qd6a"
    stage: str = "QSPIN-PROD"
    enabled: bool = False
    shadow_only: bool = True
    checkpoints: Tuple[str, ...] = (
        "source_matrix_complete",
        "kill_switch_enabled",
        "rollback_evidence_present",
        "commit_gate_approval",
        "runtime_shadow_only",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinProductionPlan":
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if self.enabled:
            raise ValueError("QSPIN production plan must remain disabled")
        if not self.shadow_only:
            raise ValueError("QSPIN production plan must remain shadow-only")
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stage": self.stage,
            "enabled": self.enabled,
            "shadow_only": self.shadow_only,
            "checkpoints": list(self.checkpoints),
            "metadata": dict(self.metadata),
        }


def build_default_qspin_production_plan() -> QSpinProductionPlan:
    return QSpinProductionPlan(
        metadata={
            "mode": "metadata_only",
            "runtime_activation": "deferred",
        }
    ).validate()
