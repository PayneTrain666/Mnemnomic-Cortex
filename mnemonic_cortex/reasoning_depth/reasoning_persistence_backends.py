from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class PersistenceBackendError(ValueError):
    """Raised when a persistence backend stub is configured unsafely."""


class PersistenceBackendKind(str, Enum):
    IN_MEMORY_DRY_RUN = "in_memory_dry_run"
    FILE_STUB = "file_stub"
    VECTOR_STORE_STUB = "vector_store_stub"
    GRAPH_STORE_STUB = "graph_store_stub"
    EXTERNAL_DISABLED = "external_disabled"


class BackendHealthStatus(str, Enum):
    DISABLED = "disabled"
    DRY_RUN_READY = "dry_run_ready"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class PersistenceBackendConfig:
    """Backend stub config.

    This module intentionally provides backend *stubs* and dependency checks.
    It does not open files, databases, vector stores, graph stores, or external
    services. It is a readiness layer for a later explicitly-approved backend.
    """

    enabled: bool = False
    backend_kind: PersistenceBackendKind = PersistenceBackendKind.IN_MEMORY_DRY_RUN
    backend_name: str = "reasoning_backend_stub"
    allow_real_writes: bool = False
    require_dry_run: bool = True
    require_dependency_check: bool = True
    max_payload_items: int = 128
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.backend_name:
            raise PersistenceBackendError("backend_name is required")
        if self.allow_real_writes:
            raise PersistenceBackendError("real backend writes are not allowed in REASON-4B")
        if not self.require_dry_run:
            raise PersistenceBackendError("dry-run mode must be required")
        if self.max_payload_items <= 0:
            raise PersistenceBackendError("max_payload_items must be positive")
        if not self.no_mutation_by_default:
            raise PersistenceBackendError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "PersistenceBackendConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "PersistenceBackendConfig":
        return cls(enabled=True)


@dataclass
class BackendDependencyReport:
    """JSON-safe dependency/readiness report for a backend stub."""

    backend_name: str
    backend_kind: PersistenceBackendKind
    status: BackendHealthStatus
    checks: Dict[str, Any] = field(default_factory=dict)
    blocked_actions: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"backend_dependency_report_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "backend_name": self.backend_name,
            "backend_kind": self.backend_kind.value,
            "status": self.status.value,
            "checks": _safe_jsonable(self.checks),
            "blocked_actions": list(self.blocked_actions),
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class BackendDryRunWriteResult:
    """Result of a dry-run backend write.

    This is explicitly not a persistence write. It records whether the payload
    would be accepted by the stub bounds and safety rules.
    """

    accepted: bool
    backend_name: str
    payload_id: str
    reason: str
    dependency_report: Dict[str, Any]
    real_write_performed: bool = False
    blocked_actions: List[str] = field(default_factory=list)
    result_id: str = field(default_factory=lambda: f"backend_dry_run_result_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "accepted": bool(self.accepted),
            "backend_name": self.backend_name,
            "payload_id": self.payload_id,
            "reason": self.reason,
            "dependency_report": _safe_jsonable(self.dependency_report),
            "real_write_performed": bool(self.real_write_performed),
            "blocked_actions": list(self.blocked_actions),
        }
        json.dumps(payload, sort_keys=True)
        return payload


class PersistenceBackendStub:
    """Dry-run-only backend stub.

    It validates payload metadata and dependency readiness without writing to
    any backing store.
    """

    def __init__(self, config: Optional[PersistenceBackendConfig] = None):
        self.config = config or PersistenceBackendConfig.disabled()
        self.config.validate()

    def dependency_check(self, *, lineage: Optional[Dict[str, Any]] = None) -> BackendDependencyReport:
        if not self.config.enabled:
            status = BackendHealthStatus.DISABLED
        elif self.config.backend_kind == PersistenceBackendKind.EXTERNAL_DISABLED:
            status = BackendHealthStatus.BLOCKED
        else:
            status = BackendHealthStatus.DRY_RUN_READY

        return BackendDependencyReport(
            backend_name=self.config.backend_name,
            backend_kind=self.config.backend_kind,
            status=status,
            checks={
                "enabled": bool(self.config.enabled),
                "dry_run_required": bool(self.config.require_dry_run),
                "allow_real_writes": bool(self.config.allow_real_writes),
                "dependency_check_metadata_only": True,
                "external_connection_opened": False,
            },
            blocked_actions=[
                "real_backend_write",
                "external_service_connection",
                "file_mutation",
                "database_mutation",
                "vector_store_mutation",
                "graph_store_mutation",
            ],
            lineage=lineage or {},
        )

    def dry_run_write(self, payload: Dict[str, Any], *, lineage: Optional[Dict[str, Any]] = None) -> BackendDryRunWriteResult:
        report = self.dependency_check(lineage=lineage).to_dict()
        payload_id = str(payload.get("payload_id") or payload.get("id") or "unknown_payload")
        items = payload.get("items", [])
        if not self.config.enabled:
            return BackendDryRunWriteResult(False, self.config.backend_name, payload_id, "backend disabled", report, blocked_actions=["backend_disabled"])
        if report["status"] != BackendHealthStatus.DRY_RUN_READY.value:
            return BackendDryRunWriteResult(False, self.config.backend_name, payload_id, "backend not dry-run ready", report, blocked_actions=["backend_not_ready"])
        if not isinstance(items, list):
            return BackendDryRunWriteResult(False, self.config.backend_name, payload_id, "payload items must be a list", report, blocked_actions=["malformed_payload"])
        if len(items) > self.config.max_payload_items:
            return BackendDryRunWriteResult(False, self.config.backend_name, payload_id, "payload exceeds max_payload_items", report, blocked_actions=["payload_too_large"])

        return BackendDryRunWriteResult(True, self.config.backend_name, payload_id, "dry-run accepted; no real write performed", report)


def reasoning_persistence_backends_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_persistence_backends",
        "stage": "REASON-4B",
        "default_enabled": False,
        "backend_stubs_only": True,
        "dry_run_only": True,
        "real_write_performed": False,
        "external_connection_opened": False,
        "json_safe_reports": True,
    }
