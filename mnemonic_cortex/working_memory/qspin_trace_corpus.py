"""QSPIN-PROD-6 deterministic trace corpus.

Metadata-only trace corpus builder and replay verifier for synthetic/sandbox-only
PROD-6 stress testing. It never stores raw payloads, secrets, tensors, or live
runtime data.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import hashlib, json

class TraceCorpusMode(str, Enum):
    SYNTHETIC_ONLY = "synthetic_only"
    DISABLED = "disabled"

class TraceCorpusStatus(str, Enum):
    BUILT = "built"
    REPLAY_PASSED = "replay_passed"
    REPLAY_FAILED = "replay_failed"
    STRUCTURED_SKIP = "structured_skip"

class TraceCorpusRecordKind(str, Enum):
    ROUTE_SHAPE_CANARY = "route_shape_canary"
    PAYLOAD_INTEGRITY_CANARY = "payload_integrity_canary"
    QH_NO_WRITE_CANARY = "qh_no_write_canary"
    SHARED_SLOT_NO_WRITE_CANARY = "shared_slot_no_write_canary"
    EXTERNAL_MEMORY_NO_WRITE_CANARY = "external_memory_no_write_canary"
    TOPOLOGY_NO_EXECUTE_CANARY = "topology_no_execute_canary"
    COMMIT_GATE_NO_COMMIT_CANARY = "commit_gate_no_commit_canary"
    AUDIT_CHAIN_CANARY = "audit_chain_canary"
    REDACTION_CANARY = "redaction_canary"
    CONCURRENCY_GUARD_CANARY = "concurrency_guard_canary"
    TIMEOUT_BOUND_CANARY = "timeout_bound_canary"
    MALFORMED_INPUT_CANARY = "malformed_input_canary"
    SYNTHETIC_PAYLOAD_HARNESS_EVENT = "synthetic_payload_harness_event"
    SYNTHETIC_SANDBOX_EVENT = "synthetic_sandbox_event"
    EXPANDED_COMMIT_GATE_EVENT = "expanded_commit_gate_event"
    ACTIVE_DRY_RUN_EXECUTOR_EVENT = "active_dry_run_executor_event"
    SAFETY_REGRESSION_EVENT = "safety_regression_event"
    SKIP_DEGRADATION_EVENT = "skip_degradation_event"
    REMEDIATION_EVENT = "remediation_event"

_REQUIRED_KINDS = tuple(TraceCorpusRecordKind)

@dataclass(frozen=True)
class TraceCorpusRecord:
    kind: TraceCorpusRecordKind
    logical_id: str
    safe_summary: Mapping[str, Any] = field(default_factory=dict)
    lineage: Tuple[str, ...] = ("QSPIN-PROD-6-QD6A",)

    def validate(self) -> "TraceCorpusRecord":
        if not isinstance(self.kind, TraceCorpusRecordKind):
            raise ValueError("invalid trace corpus kind")
        if not self.logical_id:
            raise ValueError("logical_id is required")
        summary = json.dumps(dict(self.safe_summary), sort_keys=True)
        lowered = summary.lower()
        for forbidden in ("raw_payload", "secret", "token=", "api_key", "password"):
            if forbidden in lowered:
                raise ValueError(f"unsafe trace corpus summary contains {forbidden}")
        return self

    @property
    def record_id(self) -> str:
        payload = f"{self.kind.value}|{self.logical_id}|{json.dumps(dict(self.safe_summary), sort_keys=True)}|{'/'.join(self.lineage)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def stable_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self, include_hash: bool = True) -> Dict[str, Any]:
        data = {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "logical_id": self.logical_id,
            "safe_summary": dict(self.safe_summary),
            "lineage": list(self.lineage),
        }
        if include_hash:
            data["stable_hash"] = self.stable_hash()
        return data

@dataclass(frozen=True)
class TraceCorpusBuildConfig:
    mode: TraceCorpusMode = TraceCorpusMode.SYNTHETIC_ONLY
    seed: int = 606
    include_required_records: bool = True
    max_records: int = 256

    def validate(self) -> "TraceCorpusBuildConfig":
        if self.mode is TraceCorpusMode.DISABLED:
            raise ValueError("trace corpus disabled")
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        return self

@dataclass(frozen=True)
class TraceCorpusBuildResult:
    status: TraceCorpusStatus
    records: Tuple[TraceCorpusRecord, ...]
    corpus_hash: str
    coverage: Mapping[str, bool]

    def validate(self) -> "TraceCorpusBuildResult":
        for record in self.records:
            record.validate()
        if self.status is TraceCorpusStatus.BUILT and not self.records:
            raise ValueError("built corpus requires records")
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "corpus_hash": self.corpus_hash, "records": [r.to_dict() for r in self.records], "coverage": dict(self.coverage)}

    def to_markdown(self) -> str:
        lines = ["# QSPIN-PROD-6 Trace Corpus", "", f"Status: `{self.status.value}`", f"Corpus hash: `{self.corpus_hash}`", "", "| Kind | Covered |", "|---|---:|"]
        for k, v in sorted(self.coverage.items()):
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines) + "\n"

@dataclass(frozen=True)
class TraceCorpusReplayResult:
    status: TraceCorpusStatus
    expected_hash: str
    actual_hash: str
    mismatch_count: int = 0

    @property
    def passed(self) -> bool:
        return self.status is TraceCorpusStatus.REPLAY_PASSED and self.expected_hash == self.actual_hash and self.mismatch_count == 0

class DeterministicTraceCorpusBuilder:
    def __init__(self, config: TraceCorpusBuildConfig | None = None):
        self.config = (config or build_default_trace_corpus_config()).validate()

    def build(self) -> TraceCorpusBuildResult:
        records = []
        if self.config.include_required_records:
            for idx, kind in enumerate(_REQUIRED_KINDS):
                records.append(TraceCorpusRecord(kind, f"{kind.value}_{self.config.seed}_{idx}", {
                    "synthetic": True,
                    "sandbox": True,
                    "dry_run": True,
                    "sequence": idx,
                    "safe": True,
                }))
        records = tuple(sorted(records, key=lambda r: (r.kind.value, r.logical_id)))[: self.config.max_records]
        payload = [r.to_dict(include_hash=False) for r in records]
        corpus_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        coverage = {kind.value: any(r.kind is kind for r in records) for kind in _REQUIRED_KINDS}
        return TraceCorpusBuildResult(TraceCorpusStatus.BUILT, records, corpus_hash, coverage).validate()

    def replay(self, build_result: TraceCorpusBuildResult) -> TraceCorpusReplayResult:
        build_result.validate()
        payload = [r.to_dict(include_hash=False) for r in build_result.records]
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        status = TraceCorpusStatus.REPLAY_PASSED if actual == build_result.corpus_hash else TraceCorpusStatus.REPLAY_FAILED
        return TraceCorpusReplayResult(status, build_result.corpus_hash, actual, 0 if actual == build_result.corpus_hash else 1)


def build_default_trace_corpus_config() -> TraceCorpusBuildConfig:
    return TraceCorpusBuildConfig().validate()
