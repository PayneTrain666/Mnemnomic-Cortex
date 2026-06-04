from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class BackendInterfaceError(ValueError):
    """Raised when backend interface configuration or payloads are unsafe."""


class BackendInterfaceKind(str, Enum):
    JSONL_SHADOW = "jsonl_shadow"
    SQLITE_SHADOW = "sqlite_shadow"


class BackendWriteMode(str, Enum):
    DRY_RUN_ONLY = "dry_run_only"
    REAL_WRITE_BLOCKED = "real_write_blocked"


@dataclass(frozen=True)
class BackendInterfaceConfig:
    """Dry-run-first backend interface config.

    This is an interface layer only. It is intentionally incapable of real
    writes unless a later separate write-permission stage replaces the blocked
    mode with a reviewed implementation.
    """

    enabled: bool = False
    backend_kind: BackendInterfaceKind = BackendInterfaceKind.JSONL_SHADOW
    write_mode: BackendWriteMode = BackendWriteMode.DRY_RUN_ONLY
    allow_real_writes: bool = False
    require_idempotency_key: bool = True
    require_audit_metadata: bool = True
    require_json_safe_payload: bool = True
    require_redaction_status: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.no_mutation_by_default:
            raise BackendInterfaceError("no_mutation_by_default must remain true")
        if self.allow_real_writes:
            raise BackendInterfaceError("real writes are not allowed in REAL-BACKEND-IMPLEMENTATION-A")
        if self.write_mode != BackendWriteMode.DRY_RUN_ONLY:
            raise BackendInterfaceError("write_mode must remain dry_run_only")
        if not self.require_idempotency_key:
            raise BackendInterfaceError("idempotency key is required")

    @classmethod
    def disabled(cls) -> "BackendInterfaceConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "BackendInterfaceConfig":
        return cls(enabled=True)


@dataclass
class BackendPayloadEnvelope:
    payload_kind: str
    payload: Dict[str, Any]
    idempotency_key: str
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    redaction_status: str = "not_required"
    envelope_id: str = field(default_factory=lambda: f"backend_payload_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        if not self.idempotency_key:
            raise BackendInterfaceError("idempotency_key is required")
        payload = {
            "envelope_id": self.envelope_id,
            "payload_kind": self.payload_kind,
            "payload": _safe_jsonable(self.payload),
            "idempotency_key": self.idempotency_key,
            "audit_metadata": _safe_jsonable(self.audit_metadata),
            "redaction_status": self.redaction_status,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class BackendDryRunResult:
    accepted: bool
    dry_run: bool
    backend_kind: BackendInterfaceKind
    idempotency_key: str
    would_write_payload: Dict[str, Any]
    blocked_actions: List[str] = field(default_factory=list)
    result_id: str = field(default_factory=lambda: f"backend_dry_run_result_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "accepted": bool(self.accepted),
            "dry_run": bool(self.dry_run),
            "backend_kind": self.backend_kind.value,
            "idempotency_key": self.idempotency_key,
            "would_write_payload": _safe_jsonable(self.would_write_payload),
            "blocked_actions": list(self.blocked_actions),
            "real_store_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@runtime_checkable
class BackendInterfaceProtocol(Protocol):
    config: BackendInterfaceConfig

    def dry_run_write(self, envelope: BackendPayloadEnvelope) -> BackendDryRunResult:
        """Return a JSON-safe dry-run result without writing to any store."""
        ...


def backend_interface_protocol_contract() -> Dict[str, Any]:
    return {
        "module": "backend_interface_protocol",
        "stage": "REAL-BACKEND-IMPLEMENTATION-A",
        "dry_run_first": True,
        "real_store_write_performed": False,
        "real_writes_allowed": False,
        "requires_idempotency_key": True,
        "json_safe_payloads": True,
    }
