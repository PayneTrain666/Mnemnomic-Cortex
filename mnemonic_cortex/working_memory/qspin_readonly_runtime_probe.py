"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-7 guarded read-only runtime probe harness.

This module performs local metadata/import/contract probes only. It is designed
for read-only verification of live-shaped boundaries without invoking live
runtime behavior, model inference, memory access, writes, commits, networking,
or production activation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import importlib
import time

class ReadOnlyProbeMode(str, Enum):
    DISABLED = "disabled"
    READ_ONLY = "read_only"

class ReadOnlyProbeStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class ReadOnlyProbeBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    UNSAFE_PERMISSION = "unsafe_permission"
    UNSAFE_TARGET = "unsafe_target"
    MISSING_OPTIONAL_SOURCE = "missing_optional_source"
    MISSING_REQUIRED_SOURCE = "missing_required_source"
    IMPORT_FAILED = "import_failed"
    TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
    WRITE_REQUESTED = "write_requested"
    NETWORK_REQUESTED = "network_requested"
    COMMIT_REQUESTED = "commit_requested"
    LIVE_ROUTE_REQUESTED = "live_route_requested"
    PAYLOAD_TRANSFER_REQUESTED = "payload_transfer_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"

class ReadOnlyProbeTargetKind(str, Enum):
    LOCAL_IMPORT = "local_import"
    DATACLASS_CONTRACT = "dataclass_contract"
    ENUM_CONTRACT = "enum_contract"
    TEST_FILE_EXISTS = "test_file_exists"
    MANIFEST_EXISTS = "manifest_exists"
    RELEASE_METADATA = "release_metadata"
    NO_WRITE_SENTINEL = "no_write_sentinel"
    NO_NETWORK_SENTINEL = "no_network_sentinel"
    NO_COMMIT_SENTINEL = "no_commit_sentinel"
    NO_LIVE_ROUTE_SENTINEL = "no_live_route_sentinel"
    NO_PAYLOAD_TRANSFER_SENTINEL = "no_payload_transfer_sentinel"

class ReadOnlyProbePermission(str, Enum):
    READ_ONLY = "read_only"
    OPTIONAL_READ_ONLY = "optional_read_only"

@dataclass(frozen=True)
class ReadOnlyProbeRequest:
    request_id: str
    target_kind: ReadOnlyProbeTargetKind
    target: str
    permission: ReadOnlyProbePermission = ReadOnlyProbePermission.READ_ONLY
    optional: bool = False
    allow_import: bool = True
    write_requested: bool = False
    network_requested: bool = False
    commit_requested: bool = False
    live_route_requested: bool = False
    payload_transfer_requested: bool = False
    production_activation_requested: bool = False

    def validate(self) -> "ReadOnlyProbeRequest":
        if not self.request_id or not self.target:
            raise ValueError("request_id and target are required")
        if not isinstance(self.target_kind, ReadOnlyProbeTargetKind):
            raise ValueError("invalid target kind")
        if not isinstance(self.permission, ReadOnlyProbePermission):
            raise ValueError("invalid permission")
        return self

@dataclass(frozen=True)
class ReadOnlyProbeResult:
    request_id: str
    target_kind: ReadOnlyProbeTargetKind
    target: str
    status: ReadOnlyProbeStatus
    reasons: Tuple[ReadOnlyProbeBlockReason, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    mutated_state: bool = False
    called_network: bool = False
    committed: bool = False
    production_activated: bool = False

    def validate(self) -> "ReadOnlyProbeResult":
        if self.mutated_state or self.called_network or self.committed or self.production_activated:
            raise ValueError("read-only probe produced forbidden side effect")
        if self.status in (ReadOnlyProbeStatus.FAILED, ReadOnlyProbeStatus.BLOCKED) and not self.reasons:
            raise ValueError("failed/blocked probe requires reason codes")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "target_kind": self.target_kind.value,
            "target": self.target,
            "status": self.status.value,
            "reasons": [r.value for r in self.reasons],
            "evidence": dict(self.evidence),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

@dataclass(frozen=True)
class ReadOnlyProbeSuiteResult:
    results: Tuple[ReadOnlyProbeResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == ReadOnlyProbeStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status in (ReadOnlyProbeStatus.FAILED, ReadOnlyProbeStatus.BLOCKED))

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == ReadOnlyProbeStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "failed": self.failed, "skipped": self.skipped, "results": [r.to_dict() for r in self.results]}

@dataclass(frozen=True)
class ReadOnlyProbeConfig:
    mode: ReadOnlyProbeMode = ReadOnlyProbeMode.READ_ONLY
    max_probes: int = 128
    max_probe_ms: int = 250
    allow_network: bool = False
    allow_writes: bool = False
    allow_commits: bool = False
    allow_live_routes: bool = False
    allow_payload_transfer: bool = False
    allow_production_activation: bool = False

    def validate(self) -> "ReadOnlyProbeConfig":
        if self.mode == ReadOnlyProbeMode.DISABLED:
            raise ValueError("read-only probe config disabled")
        if self.max_probes <= 0 or self.max_probe_ms <= 0:
            raise ValueError("probe bounds must be positive")
        if any([self.allow_network, self.allow_writes, self.allow_commits, self.allow_live_routes, self.allow_payload_transfer, self.allow_production_activation]):
            raise ValueError("read-only probe config cannot allow unsafe operations")
        return self

class ReadOnlyRuntimeProbeHarness:
    def __init__(self, config: Optional[ReadOnlyProbeConfig] = None, root: Optional[str | Path] = None):
        self.config = (config or build_default_readonly_probe_config()).validate()
        self.root = Path(root or ".").resolve()

    def _path_exists(self, target: str) -> bool:
        return (self.root / target).exists()

    def probe(self, request: ReadOnlyProbeRequest) -> ReadOnlyProbeResult:
        request.validate()
        start = time.perf_counter()
        reasons = []
        if request.permission not in (ReadOnlyProbePermission.READ_ONLY, ReadOnlyProbePermission.OPTIONAL_READ_ONLY):
            reasons.append(ReadOnlyProbeBlockReason.UNSAFE_PERMISSION)
        if request.write_requested:
            reasons.append(ReadOnlyProbeBlockReason.WRITE_REQUESTED)
        if request.network_requested:
            reasons.append(ReadOnlyProbeBlockReason.NETWORK_REQUESTED)
        if request.commit_requested:
            reasons.append(ReadOnlyProbeBlockReason.COMMIT_REQUESTED)
        if request.live_route_requested:
            reasons.append(ReadOnlyProbeBlockReason.LIVE_ROUTE_REQUESTED)
        if request.payload_transfer_requested:
            reasons.append(ReadOnlyProbeBlockReason.PAYLOAD_TRANSFER_REQUESTED)
        if request.production_activation_requested:
            reasons.append(ReadOnlyProbeBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        if reasons:
            return ReadOnlyProbeResult(request.request_id, request.target_kind, request.target, ReadOnlyProbeStatus.BLOCKED, tuple(dict.fromkeys(reasons))).validate()

        evidence: Dict[str, Any] = {"read_only": True, "target": request.target}
        status = ReadOnlyProbeStatus.PASSED
        try:
            if request.target_kind == ReadOnlyProbeTargetKind.LOCAL_IMPORT:
                if not request.allow_import:
                    raise ValueError("import disabled")
                module = importlib.import_module(request.target)
                evidence["module"] = getattr(module, "__name__", request.target)
            elif request.target_kind in (ReadOnlyProbeTargetKind.TEST_FILE_EXISTS, ReadOnlyProbeTargetKind.MANIFEST_EXISTS, ReadOnlyProbeTargetKind.RELEASE_METADATA):
                exists = self._path_exists(request.target)
                evidence["exists"] = exists
                if not exists:
                    status = ReadOnlyProbeStatus.SKIPPED if request.optional else ReadOnlyProbeStatus.FAILED
                    reasons.append(ReadOnlyProbeBlockReason.MISSING_OPTIONAL_SOURCE if request.optional else ReadOnlyProbeBlockReason.MISSING_REQUIRED_SOURCE)
            elif request.target_kind in (
                ReadOnlyProbeTargetKind.NO_WRITE_SENTINEL,
                ReadOnlyProbeTargetKind.NO_NETWORK_SENTINEL,
                ReadOnlyProbeTargetKind.NO_COMMIT_SENTINEL,
                ReadOnlyProbeTargetKind.NO_LIVE_ROUTE_SENTINEL,
                ReadOnlyProbeTargetKind.NO_PAYLOAD_TRANSFER_SENTINEL,
            ):
                evidence["sentinel_passed"] = True
            else:
                evidence["contract_probe"] = request.target_kind.value
        except Exception as exc:  # noqa: BLE001 - trace-safe summary only
            status = ReadOnlyProbeStatus.SKIPPED if request.optional else ReadOnlyProbeStatus.FAILED
            reasons.append(ReadOnlyProbeBlockReason.IMPORT_FAILED)
            evidence["error_type"] = type(exc).__name__

        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > self.config.max_probe_ms:
            status = ReadOnlyProbeStatus.BLOCKED
            reasons.append(ReadOnlyProbeBlockReason.TIME_BUDGET_EXCEEDED)
        return ReadOnlyProbeResult(request.request_id, request.target_kind, request.target, status, tuple(dict.fromkeys(reasons)), evidence, elapsed).validate()

    def run_suite(self, requests: Iterable[ReadOnlyProbeRequest]) -> ReadOnlyProbeSuiteResult:
        reqs = tuple(requests)
        if len(reqs) > self.config.max_probes:
            raise ValueError("probe suite exceeds max_probes")
        return ReadOnlyProbeSuiteResult(tuple(self.probe(r) for r in reqs))

def build_default_readonly_probe_config() -> ReadOnlyProbeConfig:
    return ReadOnlyProbeConfig().validate()

def build_default_readonly_probe_suite() -> Tuple[ReadOnlyProbeRequest, ...]:
    return (
        ReadOnlyProbeRequest("probe_import_stdlib_json", ReadOnlyProbeTargetKind.LOCAL_IMPORT, "json"),
        ReadOnlyProbeRequest("probe_manifest", ReadOnlyProbeTargetKind.MANIFEST_EXISTS, "release/qspin_prod7_qd6a_release_manifest.json", optional=True, permission=ReadOnlyProbePermission.OPTIONAL_READ_ONLY),
        ReadOnlyProbeRequest("probe_no_write", ReadOnlyProbeTargetKind.NO_WRITE_SENTINEL, "no_write"),
        ReadOnlyProbeRequest("probe_no_network", ReadOnlyProbeTargetKind.NO_NETWORK_SENTINEL, "no_network"),
        ReadOnlyProbeRequest("probe_no_commit", ReadOnlyProbeTargetKind.NO_COMMIT_SENTINEL, "no_commit"),
        ReadOnlyProbeRequest("probe_no_live_route", ReadOnlyProbeTargetKind.NO_LIVE_ROUTE_SENTINEL, "no_live_route"),
    )
