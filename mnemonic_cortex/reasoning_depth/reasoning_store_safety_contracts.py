"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning store safety contracts.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid


class StoreSafetyError(ValueError):
    """Raised when store-safety contracts are malformed or unsafe."""


class StoreOperationKind(str, Enum):
    DRY_RUN = "dry_run"
    PROPOSE = "propose"
    COMMIT = "commit"
    ROLLBACK = "rollback"


class StoreSafetyLevel(str, Enum):
    SAFE_METADATA_ONLY = "safe_metadata_only"
    EXPLICIT_WRITE_REQUIRED = "explicit_write_required"
    BLOCKED = "blocked"


def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _safe_jsonable(value.to_dict())
    return str(value)


@dataclass(frozen=True)
class StoreSafetyContractConfig:
    """Config for explicit store-safety contract generation.

    The contract defaults to metadata-only. Permanent persistence remains
    blocked unless a later stage wires a real store backend and explicit
    write permission.
    """

    enabled: bool = False
    allow_commit_intent: bool = False
    require_write_permission: bool = True
    require_dry_run_first: bool = True
    require_idempotency_key: bool = True
    require_audit_metadata: bool = True
    block_model_weight_mutation: bool = True
    block_optimizer_mutation: bool = True
    block_destructive_replacement: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise StoreSafetyError("no_mutation_by_default must remain true")
        if not self.require_write_permission:
            raise StoreSafetyError("write permission must be required")
        if not self.block_model_weight_mutation or not self.block_optimizer_mutation:
            raise StoreSafetyError("model/optimizer mutation must remain blocked")

    @classmethod
    def disabled(cls) -> "StoreSafetyContractConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "StoreSafetyContractConfig":
        return cls(enabled=True)


@dataclass
class StoreSafetyContract:
    """Serializable store-safety contract for persistence/commit operations."""

    enabled: bool
    operation_kind: StoreOperationKind
    safety_level: StoreSafetyLevel
    write_permission_required: bool = True
    dry_run_first_required: bool = True
    idempotency_key_required: bool = True
    audit_metadata_required: bool = True
    blocked_actions: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    contract_id: str = field(default_factory=lambda: f"store_safety_contract_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "contract_id": self.contract_id,
            "enabled": bool(self.enabled),
            "operation_kind": self.operation_kind.value,
            "safety_level": self.safety_level.value,
            "write_permission_required": bool(self.write_permission_required),
            "dry_run_first_required": bool(self.dry_run_first_required),
            "idempotency_key_required": bool(self.idempotency_key_required),
            "audit_metadata_required": bool(self.audit_metadata_required),
            "blocked_actions": list(self.blocked_actions),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "audit_metadata": True,
                "write_permission_hooks": True,
                "conflict_hooks": True,
                "quarantine_hooks": True,
                "policy_lane_integration": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class StoreSafetyContractBuilder:
    """Builds explicit store-safety metadata without opening a store backend."""

    def __init__(self, config: Optional[StoreSafetyContractConfig] = None):
        self.config = config or StoreSafetyContractConfig.disabled()
        self.config.validate()

    def build(
        self,
        *,
        operation_kind: StoreOperationKind = StoreOperationKind.DRY_RUN,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> StoreSafetyContract:
        blocked_actions = [
            "automatic_memory_store_write",
            "automatic_model_weight_mutation",
            "automatic_optimizer_mutation",
            "destructive_wm_mann_ltm_replacement",
            "hidden_component_activation",
            "fake_production_complete_claim",
        ]
        if operation_kind == StoreOperationKind.COMMIT and not self.config.allow_commit_intent:
            safety_level = StoreSafetyLevel.BLOCKED
            blocked_actions.append("commit_intent_not_enabled")
        elif operation_kind == StoreOperationKind.COMMIT:
            safety_level = StoreSafetyLevel.EXPLICIT_WRITE_REQUIRED
        else:
            safety_level = StoreSafetyLevel.SAFE_METADATA_ONLY

        return StoreSafetyContract(
            enabled=bool(self.config.enabled),
            operation_kind=operation_kind,
            safety_level=safety_level,
            write_permission_required=self.config.require_write_permission,
            dry_run_first_required=self.config.require_dry_run_first,
            idempotency_key_required=self.config.require_idempotency_key,
            audit_metadata_required=self.config.require_audit_metadata,
            blocked_actions=blocked_actions,
            safety_flags={
                "permanent_memory_store_mutation": False,
                "automatic_persistence": False,
                "model_weight_mutation": False,
                "optimizer_mutation": False,
                "destructive_replacement": False,
                "commit_intent_allowed": bool(self.config.allow_commit_intent),
            },
            lineage=lineage or {},
        )


def reasoning_store_safety_contracts_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_store_safety_contracts",
        "stage": "REASON-4A",
        "default_enabled": False,
        "metadata_only": True,
        "write_permission_required": True,
        "automatic_persistence": False,
        "permanent_memory_store_mutation": False,
        "json_safe_contract": True,
    }
