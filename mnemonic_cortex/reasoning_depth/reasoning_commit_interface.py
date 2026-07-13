"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning commit interface.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid

from .reasoning_store_safety_contracts import (
    StoreOperationKind,
    StoreSafetyContractBuilder,
    StoreSafetyContractConfig,
    _safe_jsonable,
)


class ReasoningCommitInterfaceError(ValueError):
    """Raised when commit request/decision handling is unsafe."""


class CommitDecisionStatus(str, Enum):
    DISABLED = "disabled"
    DRY_RUN_ONLY = "dry_run_only"
    DENIED = "denied"
    APPROVED_METADATA_ONLY = "approved_metadata_only"
    APPROVED_EXPLICIT_WRITE_INTENT = "approved_explicit_write_intent"


@dataclass(frozen=True)
class ReasoningCommitInterfaceConfig:
    """Explicit commit interface config.

    Default behavior denies permanent writes. `allow_explicit_commit=True` only
    approves a write-intent decision object; it still performs no real write.
    """

    enabled: bool = False
    allow_explicit_commit: bool = False
    require_write_permission: bool = True
    require_idempotency_key: bool = True
    require_dry_run: bool = True
    max_payload_items: int = 128
    require_json_safe_payload: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise ReasoningCommitInterfaceError("no_mutation_by_default must remain true")
        if not self.require_write_permission:
            raise ReasoningCommitInterfaceError("write permission must be required")
        if self.max_payload_items <= 0:
            raise ReasoningCommitInterfaceError("max_payload_items must be positive")

    @classmethod
    def disabled(cls) -> "ReasoningCommitInterfaceConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningCommitInterfaceConfig":
        return cls(enabled=True)


@dataclass
class ReasoningCommitRequest:
    """JSON-safe explicit commit request payload."""

    target_store: str
    payload: Dict[str, Any]
    operation_kind: StoreOperationKind = StoreOperationKind.PROPOSE
    write_permission: bool = False
    dry_run: bool = True
    idempotency_key: str = ""
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"reasoning_commit_request_{uuid.uuid4().hex[:16]}")

    def __post_init__(self) -> None:
        if not self.target_store:
            raise ReasoningCommitInterfaceError("target_store is required")
        if not isinstance(self.payload, dict):
            raise ReasoningCommitInterfaceError("payload must be a dict")
        if not self.idempotency_key:
            digest = hashlib.sha256(
                json.dumps(_safe_jsonable({"target_store": self.target_store, "payload": self.payload}), sort_keys=True).encode("utf-8")
            ).hexdigest()[:24]
            object.__setattr__(self, "idempotency_key", f"commit_{digest}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "target_store": self.target_store,
            "payload": _safe_jsonable(self.payload),
            "operation_kind": self.operation_kind.value,
            "write_permission": bool(self.write_permission),
            "dry_run": bool(self.dry_run),
            "idempotency_key": self.idempotency_key,
            "audit_metadata": _safe_jsonable(self.audit_metadata),
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class ReasoningCommitDecision:
    """Decision object returned by the commit interface; it never performs writes."""

    status: CommitDecisionStatus
    approved: bool
    reason: str
    request: Dict[str, Any]
    safety_contract: Dict[str, Any]
    blocked_actions: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: f"reasoning_commit_decision_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "decision_id": self.decision_id,
            "status": self.status.value,
            "approved": bool(self.approved),
            "reason": self.reason,
            "request": _safe_jsonable(self.request),
            "safety_contract": _safe_jsonable(self.safety_contract),
            "blocked_actions": list(self.blocked_actions),
            "safety_flags": _safe_jsonable(self.safety_flags),
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningCommitInterface:
    """Validates explicit commit intent without committing to stores."""

    def __init__(self, config: Optional[ReasoningCommitInterfaceConfig] = None):
        self.config = config or ReasoningCommitInterfaceConfig.disabled()
        self.config.validate()
        self.safety_builder = StoreSafetyContractBuilder(
            StoreSafetyContractConfig(
                enabled=self.config.enabled,
                allow_commit_intent=self.config.allow_explicit_commit,
                require_write_permission=self.config.require_write_permission,
                require_dry_run_first=self.config.require_dry_run,
                require_idempotency_key=self.config.require_idempotency_key,
            )
        )

    def decide(self, request: ReasoningCommitRequest) -> ReasoningCommitDecision:
        request_payload = request.to_dict()
        contract = self.safety_builder.build(operation_kind=request.operation_kind, lineage=request.lineage).to_dict()

        if not self.config.enabled:
            return self._decision(
                CommitDecisionStatus.DISABLED,
                False,
                "commit interface disabled",
                request_payload,
                contract,
                ["interface_disabled"],
            )

        if self.config.require_json_safe_payload:
            json.dumps(request_payload, sort_keys=True)

        if len(request.payload) > self.config.max_payload_items:
            return self._decision(
                CommitDecisionStatus.DENIED,
                False,
                "payload item count exceeds configured bound",
                request_payload,
                contract,
                ["payload_too_large"],
            )

        if self.config.require_idempotency_key and not request.idempotency_key:
            return self._decision(
                CommitDecisionStatus.DENIED,
                False,
                "idempotency key missing",
                request_payload,
                contract,
                ["missing_idempotency_key"],
            )

        if self.config.require_dry_run and not request.dry_run and request.operation_kind != StoreOperationKind.COMMIT:
            return self._decision(
                CommitDecisionStatus.DENIED,
                False,
                "non-commit operations must remain dry-run",
                request_payload,
                contract,
                ["dry_run_required"],
            )

        if request.operation_kind == StoreOperationKind.COMMIT:
            if not self.config.allow_explicit_commit:
                return self._decision(
                    CommitDecisionStatus.DENIED,
                    False,
                    "explicit commit not enabled",
                    request_payload,
                    contract,
                    ["explicit_commit_disabled"],
                )
            if self.config.require_write_permission and not request.write_permission:
                return self._decision(
                    CommitDecisionStatus.DENIED,
                    False,
                    "write permission required",
                    request_payload,
                    contract,
                    ["missing_write_permission"],
                )
            return self._decision(
                CommitDecisionStatus.APPROVED_EXPLICIT_WRITE_INTENT,
                True,
                "explicit write intent approved as metadata only; no store write performed",
                request_payload,
                contract,
                [],
            )

        return self._decision(
            CommitDecisionStatus.DRY_RUN_ONLY if request.dry_run else CommitDecisionStatus.APPROVED_METADATA_ONLY,
            True,
            "metadata-only proposal accepted; no store write performed",
            request_payload,
            contract,
            [],
        )

    @staticmethod
    def _decision(
        status: CommitDecisionStatus,
        approved: bool,
        reason: str,
        request_payload: Dict[str, Any],
        contract: Dict[str, Any],
        blocked: List[str],
    ) -> ReasoningCommitDecision:
        return ReasoningCommitDecision(
            status=status,
            approved=approved,
            reason=reason,
            request=request_payload,
            safety_contract=contract,
            blocked_actions=blocked,
            safety_flags={
                "permanent_memory_store_mutation": False,
                "real_store_write_performed": False,
                "model_weight_mutation": False,
                "optimizer_mutation": False,
                "destructive_replacement": False,
            },
        )


def reasoning_commit_interface_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_commit_interface",
        "stage": "REASON-4A",
        "default_enabled": False,
        "explicit_write_permission_required": True,
        "real_store_write_performed": False,
        "permanent_memory_store_mutation": False,
        "json_safe_decision": True,
    }
