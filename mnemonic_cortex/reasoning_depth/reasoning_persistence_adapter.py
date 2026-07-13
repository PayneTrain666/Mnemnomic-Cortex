"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning persistence adapter.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid

from .reasoning_commit_interface import (
    ReasoningCommitInterface,
    ReasoningCommitInterfaceConfig,
    ReasoningCommitRequest,
)
from .reasoning_store_safety_contracts import StoreOperationKind, _safe_jsonable


class ReasoningPersistenceAdapterError(ValueError):
    """Raised when persistence adapter payload construction is unsafe."""


@dataclass(frozen=True)
class ReasoningPersistenceAdapterConfig:
    """Optional persistence adapter config.

    Disabled by default. Even when enabled, this adapter only builds payloads
    and commit-intent decisions. It never opens or mutates a real store.
    """

    enabled: bool = False
    adapter_name: str = "reasoning_persistence_adapter"
    max_items: int = 128
    allow_commit_intent: bool = False
    require_write_permission: bool = True
    require_dry_run_first: bool = True
    require_json_safe_payload: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.adapter_name:
            raise ReasoningPersistenceAdapterError("adapter_name is required")
        if self.max_items <= 0:
            raise ReasoningPersistenceAdapterError("max_items must be positive")
        if not self.no_mutation_by_default:
            raise ReasoningPersistenceAdapterError("no_mutation_by_default must remain true")
        if not self.require_write_permission:
            raise ReasoningPersistenceAdapterError("write permission must be required")

    @classmethod
    def disabled(cls) -> "ReasoningPersistenceAdapterConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningPersistenceAdapterConfig":
        return cls(enabled=True)


@dataclass
class PersistencePayload:
    """JSON-safe persistence payload for strategy graphs/traces/proposals."""

    target_store: str
    item_kind: str
    items: List[Dict[str, Any]]
    adapter_name: str
    payload_hash: str = ""
    lineage: Dict[str, Any] = field(default_factory=dict)
    payload_id: str = field(default_factory=lambda: f"persistence_payload_{uuid.uuid4().hex[:16]}")

    def __post_init__(self) -> None:
        if not self.target_store:
            raise ReasoningPersistenceAdapterError("target_store is required")
        if not self.item_kind:
            raise ReasoningPersistenceAdapterError("item_kind is required")
        if not isinstance(self.items, list):
            raise ReasoningPersistenceAdapterError("items must be a list")
        safe = _safe_jsonable({"target_store": self.target_store, "item_kind": self.item_kind, "items": self.items})
        digest = hashlib.sha256(json.dumps(safe, sort_keys=True).encode("utf-8")).hexdigest()
        object.__setattr__(self, "payload_hash", digest)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "payload_id": self.payload_id,
            "target_store": self.target_store,
            "item_kind": self.item_kind,
            "items": _safe_jsonable(self.items),
            "adapter_name": self.adapter_name,
            "payload_hash": self.payload_hash,
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class PersistenceAdapterReport:
    """Report for payload construction and optional commit-intent decision."""

    enabled: bool
    payload: Optional[Dict[str, Any]]
    commit_decision: Optional[Dict[str, Any]]
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"persistence_adapter_report_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "payload": _safe_jsonable(self.payload),
            "commit_decision": _safe_jsonable(self.commit_decision),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
            "paamax_metadata": {
                "trace_governance": True,
                "audit_metadata": True,
                "write_permission_hooks": True,
                "confidence_hook": True,
                "disagreement_hook": True,
                "conflict_hook": True,
                "quarantine_hook": True,
                "policy_lane_integration": True,
            },
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningPersistenceAdapter:
    """Builds persistence payloads without writing to storage."""

    def __init__(self, config: Optional[ReasoningPersistenceAdapterConfig] = None):
        self.config = config or ReasoningPersistenceAdapterConfig.disabled()
        self.config.validate()
        self.commit_interface = ReasoningCommitInterface(
            ReasoningCommitInterfaceConfig(
                enabled=self.config.enabled,
                allow_explicit_commit=self.config.allow_commit_intent,
                require_write_permission=self.config.require_write_permission,
                require_dry_run=self.config.require_dry_run_first,
                max_payload_items=self.config.max_items,
                require_json_safe_payload=self.config.require_json_safe_payload,
            )
        )

    def build_payload(
        self,
        *,
        target_store: str,
        item_kind: str,
        items: List[Dict[str, Any]],
        lineage: Optional[Dict[str, Any]] = None,
    ) -> PersistencePayload:
        if len(items) > self.config.max_items:
            raise ReasoningPersistenceAdapterError("items exceeds max_items")
        payload = PersistencePayload(
            target_store=target_store,
            item_kind=item_kind,
            items=items,
            adapter_name=self.config.adapter_name,
            lineage=lineage or {},
        )
        if self.config.require_json_safe_payload:
            json.dumps(payload.to_dict(), sort_keys=True)
        return payload

    def prepare(
        self,
        *,
        target_store: str,
        item_kind: str,
        items: List[Dict[str, Any]],
        operation_kind: StoreOperationKind = StoreOperationKind.PROPOSE,
        write_permission: bool = False,
        dry_run: bool = True,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> PersistenceAdapterReport:
        if not self.config.enabled:
            return PersistenceAdapterReport(
                enabled=False,
                payload=None,
                commit_decision=None,
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        payload = self.build_payload(
            target_store=target_store,
            item_kind=item_kind,
            items=items,
            lineage=lineage,
        )
        request = ReasoningCommitRequest(
            target_store=target_store,
            payload=payload.to_dict(),
            operation_kind=operation_kind,
            write_permission=write_permission,
            dry_run=dry_run,
            audit_metadata={
                "adapter_name": self.config.adapter_name,
                "payload_hash": payload.payload_hash,
                "metadata_only": True,
            },
            lineage=lineage or {},
        )
        decision = self.commit_interface.decide(request)
        return PersistenceAdapterReport(
            enabled=True,
            payload=payload.to_dict(),
            commit_decision=decision.to_dict(),
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "automatic_persistence": False,
            "permanent_memory_store_mutation": False,
            "real_store_write_performed": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
        }


def reasoning_persistence_adapter_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_persistence_adapter",
        "stage": "REASON-4A",
        "default_enabled": False,
        "automatic_persistence": False,
        "real_store_write_performed": False,
        "write_permission_required": True,
        "json_safe_payload": True,
    }
