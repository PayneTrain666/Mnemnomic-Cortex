"""HGM-7 result dataclasses for guarded write-execution scaffolding.

HGM-7 introduces a transaction log and recovery verification harness, but it
remains simulation-mode and preview-safe by default. It does not write into
QDT/WM memory unless a caller supplies an explicitly isolated test adapter and
enables test-only execution options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import hashlib

from .types import TraceRecord
from .validation import ValidationResult
from .enums import TraceEventKind, ValidationSeverity

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def hgm7_stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def hgm7_redact(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): hgm7_redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(hgm7_redact(key, v) for v in value)
    return value


def trace_hgm7(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): hgm7_redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


@dataclass(frozen=True)
class HGM7ExecutionOptions:
    """Options for guarded write-execution scaffolding.

    Defaults are intentionally safe: simulation mode enabled, test execution
    disabled, dry-run required, and commit-preview allowance required before
    any simulated/test operation can be considered structurally allowed.
    """

    simulation_mode: bool = True
    allow_test_execution: bool = False
    require_commit_preview_allowed: bool = True
    require_rollback_verified: bool = True
    max_operations: int = 128
    test_adapter_id: str = "isolated_test_adapter"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_operations) <= 0:
            raise ValueError("max_operations must be positive")
        object.__setattr__(self, "max_operations", int(self.max_operations))


@dataclass(frozen=True)
class WriteExecutionAdapterStatus:
    adapter_id: str
    available: bool
    simulation_mode: bool
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionLogEntry:
    entry_id: str
    operation_id: str
    operation_type: str
    status: str
    source_payload_id: str
    target_slot_id: str
    dry_run: bool
    simulation_mode: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionLog:
    log_id: str
    entries: Tuple[TransactionLogEntry, ...]
    complete: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class RecoveryVerificationRecord:
    verification_id: str
    rollback_id: str
    operation_id: str
    target_slot_id: str
    verified: bool
    reason: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryVerificationResult:
    records: Tuple[RecoveryVerificationRecord, ...]
    rollback_ready: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class WriteExecutionResult:
    execution_id: str
    adapter_status: WriteExecutionAdapterStatus
    transaction_log: TransactionLog
    recovery_verification: RecoveryVerificationResult
    executed: bool
    simulation_mode: bool
    allowed: bool
    blocked_reason: str
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM7WriteExecutionResult:
    execution_result: WriteExecutionResult
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
