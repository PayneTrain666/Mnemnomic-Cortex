"""HGM-10 release consolidation dataclasses.

HGM-10 freezes the additive HGM/HPME v0.1 public API surface, consolidates
release evidence, and emits an integration roadmap. It does not enable live
QDT/WM writes or mutate existing memory internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple
import hashlib

from .enums import TraceEventKind, ValidationSeverity
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def hgm10_stable_hash(*parts: Any, length: int = 16) -> str:
    """Return a deterministic short hash used for stable release IDs."""

    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def hgm10_redact(key: str, value: Any) -> Any:
    """Redact obvious secret-bearing keys in trace payloads."""

    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): hgm10_redact(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(hgm10_redact(key, v) for v in value)
    return value


def trace_hgm10(component: str, validation: ValidationResult, payload: Mapping[str, Any] | None = None) -> TraceRecord:
    """Create a trace record for HGM-10 consolidation operations."""

    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): hgm10_redact(str(k), v) for k, v in dict(payload or {}).items()},
    )


@dataclass(frozen=True)
class HGM10ReleaseOptions:
    """Options for final HGM v0.1 consolidation.

    Defaults are documentation/consolidation oriented and intentionally do not
    enable production execution, runtime memory writes, or symbol mutation.
    """

    release_version: str = "HGM-v0.1"
    max_symbols: int = 1024
    max_manifests: int = 64
    max_roadmap_items: int = 32
    api_freeze_strict: bool = True
    include_private_symbols: bool = False
    allow_live_qdt_writes: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_symbols) <= 0:
            raise ValueError("max_symbols must be positive")
        if int(self.max_manifests) <= 0:
            raise ValueError("max_manifests must be positive")
        if int(self.max_roadmap_items) <= 0:
            raise ValueError("max_roadmap_items must be positive")
        object.__setattr__(self, "max_symbols", int(self.max_symbols))
        object.__setattr__(self, "max_manifests", int(self.max_manifests))
        object.__setattr__(self, "max_roadmap_items", int(self.max_roadmap_items))


@dataclass(frozen=True)
class APIFreezeSymbol:
    symbol_id: str
    symbol_name: str
    symbol_type: str
    module_name: str
    exported: bool
    stable: bool
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol_id:
            raise ValueError("symbol_id is required")
        if not self.symbol_name:
            raise ValueError("symbol_name is required")
        if not self.module_name:
            raise ValueError("module_name is required")


@dataclass(frozen=True)
class APIFreezeRecord:
    freeze_id: str
    release_version: str
    symbols: Tuple[APIFreezeSymbol, ...]
    module_names: Tuple[str, ...]
    frozen: bool
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols or tuple()))
        object.__setattr__(self, "module_names", tuple(self.module_names or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ReleaseManifestSummary:
    summary_id: str
    stage_id: str
    manifest_path: str
    found: bool
    release_name: str
    version: str
    file_count: int
    test_summary: str
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_count", max(0, int(self.file_count)))


@dataclass(frozen=True)
class ReleaseConsolidationRecord:
    consolidation_id: str
    release_version: str
    manifest_summaries: Tuple[ReleaseManifestSummary, ...]
    documentation_files: Tuple[str, ...]
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_summaries", tuple(self.manifest_summaries or tuple()))
        object.__setattr__(self, "documentation_files", tuple(self.documentation_files or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class IntegrationRoadmapItem:
    item_id: str
    stage: str
    title: str
    priority: str
    status: str
    blocked_by: Tuple[str, ...]
    trace_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocked_by", tuple(self.blocked_by or tuple()))


@dataclass(frozen=True)
class IntegrationRoadmapRecord:
    roadmap_id: str
    release_version: str
    items: Tuple[IntegrationRoadmapItem, ...]
    next_command: str
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class HGM10ReleaseConsolidationResult:
    api_freeze: APIFreezeRecord
    release_consolidation: ReleaseConsolidationRecord
    integration_roadmap: IntegrationRoadmapRecord
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
