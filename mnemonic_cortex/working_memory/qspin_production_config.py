"""QSPIN production configuration contracts (inert-by-default)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping


class QSpinRuntimeFeatureFlag(str, Enum):
    BRIDGE_ROUTING = "bridge_routing"
    PAYLOAD_TRANSFER = "payload_transfer"
    TOPOLOGY_ROUTING = "topology_routing"
    DEPTH_PHASE_EXECUTION = "depth_phase_execution"
    SHARED_SLOT_WRITE = "shared_slot_write"
    EXTERNAL_MEMORY_WRITE = "external_memory_write"
    QH_STORAGE_WRITE = "qh_storage_write"
    COMMIT_EXECUTION = "commit_execution"
    RAW_PAYLOAD_TRACE = "raw_payload_trace"
    PRODUCTION_ACTIVATION = "production_activation"


class QSpinRuntimeKillSwitchState(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    TRIPPED = "tripped"


@dataclass(frozen=True)
class QSpinProductionConfig:
    config_id: str = "qspin_prod_config_qd6a"
    stage: str = "QSPIN-PROD"
    runtime_inert_default: bool = True
    kill_switch_default: QSpinRuntimeKillSwitchState = QSpinRuntimeKillSwitchState.ENABLED
    enabled_feature_flags: FrozenSet[QSpinRuntimeFeatureFlag] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinProductionConfig":
        if not self.config_id:
            raise ValueError("config_id is required")
        if not self.runtime_inert_default:
            raise ValueError("QSPIN production config must remain inert-by-default")
        if self.kill_switch_default is not QSpinRuntimeKillSwitchState.ENABLED:
            raise ValueError("default kill switch must be ENABLED")
        if self.enabled_feature_flags:
            raise ValueError("no QSPIN runtime feature flags may be enabled by default")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "stage": self.stage,
            "runtime_inert_default": self.runtime_inert_default,
            "kill_switch_default": self.kill_switch_default.value,
            "enabled_feature_flags": [flag.value for flag in sorted(self.enabled_feature_flags, key=lambda f: f.value)],
            "metadata": dict(self.metadata),
        }


def build_default_qspin_production_config() -> QSpinProductionConfig:
    return QSpinProductionConfig(
        metadata={
            "policy": "shadow_only",
            "no_live_routing": True,
            "no_payload_transfer": True,
            "no_writes": True,
            "no_commit_execution": True,
            "no_production_activation": True,
        }
    ).validate()
