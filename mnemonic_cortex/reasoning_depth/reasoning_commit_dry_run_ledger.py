"""
Plain-language summary
----------------------
What this file is for: Dry-run or sandbox helper (reasoning_commit_dry_run_ledger).
How it fits in the system: Lets engineers rehearse a path safely without committing live side effects.
Status: LOW-USE / SAFETY SCAFFOLD
Important notes for non-coders: Not the everyday training path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json
import time
import uuid

from .reasoning_store_safety_contracts import _safe_jsonable


class CommitDryRunLedgerError(ValueError):
    """Raised when dry-run ledger operations are unsafe."""


@dataclass(frozen=True)
class CommitDryRunLedgerConfig:
    """Config for bounded in-memory dry-run ledger.

    The ledger is in-memory and metadata-only. It does not persist to disk,
    write to stores, or mutate memory banks.
    """

    enabled: bool = False
    max_records: int = 1024
    require_idempotency: bool = True
    reject_duplicate_idempotency_key: bool = True
    require_json_safe_records: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_records <= 0:
            raise CommitDryRunLedgerError("max_records must be positive")
        if not self.no_mutation_by_default:
            raise CommitDryRunLedgerError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "CommitDryRunLedgerConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "CommitDryRunLedgerConfig":
        return cls(enabled=True)


@dataclass
class CommitDryRunLedgerRecord:
    """JSON-safe dry-run ledger record."""

    idempotency_key: str
    request: Dict[str, Any]
    decision: Dict[str, Any]
    backend_result: Optional[Dict[str, Any]] = None
    timestamp_unix: float = field(default_factory=time.time)
    lineage: Dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: f"dry_run_ledger_record_{uuid.uuid4().hex[:16]}")

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            safe = _safe_jsonable({"request": self.request, "decision": self.decision})
            digest = hashlib.sha256(json.dumps(safe, sort_keys=True).encode("utf-8")).hexdigest()[:24]
            object.__setattr__(self, "idempotency_key", f"dryrun_{digest}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "idempotency_key": self.idempotency_key,
            "request": _safe_jsonable(self.request),
            "decision": _safe_jsonable(self.decision),
            "backend_result": _safe_jsonable(self.backend_result),
            "timestamp_unix": float(self.timestamp_unix),
            "lineage": _safe_jsonable(self.lineage),
            "real_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


@dataclass
class CommitDryRunLedgerAppendResult:
    accepted: bool
    reason: str
    record: Optional[Dict[str, Any]] = None
    ledger_size: int = 0
    blocked_actions: List[str] = field(default_factory=list)
    result_id: str = field(default_factory=lambda: f"dry_run_ledger_append_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "record": _safe_jsonable(self.record),
            "ledger_size": int(self.ledger_size),
            "blocked_actions": list(self.blocked_actions),
            "real_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


class CommitDryRunLedger:
    """Bounded in-memory ledger for commit dry-runs only."""

    def __init__(self, config: Optional[CommitDryRunLedgerConfig] = None):
        self.config = config or CommitDryRunLedgerConfig.disabled()
        self.config.validate()
        self._records: List[CommitDryRunLedgerRecord] = []
        self._idempotency_keys = set()

    @property
    def records(self) -> List[CommitDryRunLedgerRecord]:
        return list(self._records)

    def append(
        self,
        *,
        request: Dict[str, Any],
        decision: Dict[str, Any],
        backend_result: Optional[Dict[str, Any]] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> CommitDryRunLedgerAppendResult:
        if not self.config.enabled:
            return CommitDryRunLedgerAppendResult(False, "ledger disabled", ledger_size=len(self._records), blocked_actions=["ledger_disabled"])

        idempotency_key = str(request.get("idempotency_key") or "")
        if self.config.require_idempotency and not idempotency_key:
            return CommitDryRunLedgerAppendResult(False, "idempotency key required", ledger_size=len(self._records), blocked_actions=["missing_idempotency_key"])
        if self.config.reject_duplicate_idempotency_key and idempotency_key in self._idempotency_keys:
            return CommitDryRunLedgerAppendResult(False, "duplicate idempotency key", ledger_size=len(self._records), blocked_actions=["duplicate_idempotency_key"])
        if len(self._records) >= self.config.max_records:
            return CommitDryRunLedgerAppendResult(False, "ledger full", ledger_size=len(self._records), blocked_actions=["ledger_full"])

        record = CommitDryRunLedgerRecord(idempotency_key, request, decision, backend_result, lineage=lineage or {})
        if self.config.require_json_safe_records:
            json.dumps(record.to_dict(), sort_keys=True)
        self._records.append(record)
        if idempotency_key:
            self._idempotency_keys.add(idempotency_key)
        return CommitDryRunLedgerAppendResult(True, "dry-run ledger record appended in memory only", record=record.to_dict(), ledger_size=len(self._records))

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "enabled": bool(self.config.enabled),
            "max_records": int(self.config.max_records),
            "record_count": len(self._records),
            "records": [record.to_dict() for record in self._records],
            "in_memory_only": True,
            "real_write_performed": False,
        }
        json.dumps(payload, sort_keys=True)
        return payload


def reasoning_commit_dry_run_ledger_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_commit_dry_run_ledger",
        "stage": "REASON-4B",
        "default_enabled": False,
        "in_memory_only": True,
        "dry_run_only": True,
        "real_write_performed": False,
        "idempotency_supported": True,
        "json_safe_records": True,
    }
