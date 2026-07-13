"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-8 CI gate baseline freeze.

Local artifact only: no CI provider configuration, no repo writes, no activation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Tuple
import json

class CIBaselineFreezeMode(str, Enum):
    LOCAL_ARTIFACT_ONLY = "local_artifact_only"

class CIBaselineFreezeStatus(str, Enum):
    FROZEN = "frozen"
    BLOCKED = "blocked"

class CIBaselineFreezeBlockReason(str, Enum):
    MISSING_CRITICAL_GATE = "missing_critical_gate"
    UNSAFE_GATE = "unsafe_gate"

class CIBaselineGateKind(str, Enum):
    SOURCE_VERIFICATION = "source_verification_gate"
    SOURCE_MATRIX = "source_consideration_matrix_gate"
    TOKEN_BUDGET = "token_budget_gate"
    NO_LIVE_ROUTING = "no_live_routing_gate"
    NO_PAYLOAD_TRANSFER = "no_payload_transfer_gate"
    NO_WRITE = "no_write_gate"
    NO_QH_WRITE = "no_qh_write_gate"
    NO_SHARED_SLOT_WRITE = "no_shared_slot_write_gate"
    NO_EXTERNAL_MEMORY_WRITE = "no_external_memory_write_gate"
    NO_COMMIT = "no_commit_gate"
    NO_PRODUCTION_ACTIVATION = "no_production_activation_gate"
    REDACTION = "redaction_gate"
    AUDIT_CHAIN = "audit_chain_gate"
    DETERMINISTIC_REPLAY = "deterministic_replay_gate"
    TRACE_CORPUS = "trace_corpus_gate"
    CI_MATRIX = "ci_matrix_gate"
    EXTENDED_SAFETY_REGRESSION = "extended_safety_regression_gate"
    READONLY_PROBE = "read_only_runtime_probe_gate"
    SYNTHETIC_REAL_BOUNDARY = "synthetic_to_real_boundary_gate"
    OBSERVABILITY_REVIEW = "observability_review_gate"
    READINESS_BLOCKER = "readiness_blocker_gate"
    FULL_PRINTOUT = "full_printout_gate"
    RELEASE_PACK = "release_pack_gate"
    SHIP_CHECK = "ship_check_gate"

@dataclass(frozen=True)
class CIBaselineGateRecord:
    kind: CIBaselineGateKind
    passed: bool
    critical: bool = True
    evidence: str = ""

    def validate(self) -> "CIBaselineGateRecord":
        if not isinstance(self.kind, CIBaselineGateKind):
            raise ValueError("kind must be CIBaselineGateKind")
        if self.critical and not self.evidence:
            raise ValueError("critical gate requires evidence")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"kind": self.kind.value, "passed": self.passed, "critical": self.critical, "evidence": self.evidence}

@dataclass(frozen=True)
class CIBaselineFreezeConfig:
    mode: CIBaselineFreezeMode = CIBaselineFreezeMode.LOCAL_ARTIFACT_ONLY
    configure_real_ci: bool = False
    commit_to_repo: bool = False

    def validate(self) -> "CIBaselineFreezeConfig":
        if self.configure_real_ci:
            raise ValueError("real CI provider configuration forbidden")
        if self.commit_to_repo:
            raise ValueError("repo commit forbidden")
        return self

@dataclass(frozen=True)
class CIBaselineFreezeRequest:
    request_id: str
    gates: Tuple[CIBaselineGateRecord, ...]

    def validate(self) -> "CIBaselineFreezeRequest":
        if not self.request_id:
            raise ValueError("request_id required")
        for gate in self.gates:
            gate.validate()
        return self

@dataclass(frozen=True)
class CIBaselineFreezeResult:
    request_id: str
    status: CIBaselineFreezeStatus
    block_reasons: Tuple[CIBaselineFreezeBlockReason, ...]
    gates: Tuple[CIBaselineGateRecord, ...]

    def to_dict(self) -> Dict[str, object]:
        return {"request_id": self.request_id, "status": self.status.value, "block_reasons": [r.value for r in self.block_reasons], "gates": [g.to_dict() for g in self.gates]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# PROD-8 CI Gate Baseline Freeze", "", f"Status: `{self.status.value}`", ""]
        for gate in self.gates:
            lines.append(f"- `{gate.kind.value}`: {'PASS' if gate.passed else 'BLOCK'}")
        return "\n".join(lines) + "\n"

class CIGateBaselineFreezer:
    def __init__(self, config: CIBaselineFreezeConfig | None = None):
        self.config = (config or build_default_ci_baseline_freeze_config()).validate()

    def freeze(self, request: CIBaselineFreezeRequest) -> CIBaselineFreezeResult:
        request.validate()
        reasons = []
        kinds = {g.kind for g in request.gates}
        missing = [k for k in CIBaselineGateKind if k not in kinds]
        if missing:
            reasons.append(CIBaselineFreezeBlockReason.MISSING_CRITICAL_GATE)
        if any(g.critical and not g.passed for g in request.gates):
            reasons.append(CIBaselineFreezeBlockReason.UNSAFE_GATE)
        status = CIBaselineFreezeStatus.BLOCKED if reasons else CIBaselineFreezeStatus.FROZEN
        return CIBaselineFreezeResult(request.request_id, status, tuple(dict.fromkeys(reasons)), request.gates)

def build_default_ci_baseline_freeze_config() -> CIBaselineFreezeConfig:
    return CIBaselineFreezeConfig().validate()

def build_default_ci_baseline_gate_records() -> Tuple[CIBaselineGateRecord, ...]:
    return tuple(CIBaselineGateRecord(kind, True, True, f"PROD-8 local baseline evidence for {kind.value}") for kind in CIBaselineGateKind)
