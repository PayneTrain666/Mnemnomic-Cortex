"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-1 rollback harness scaffolding.

Dry-run only. It validates rollback evidence and does not delete files or
mutate runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple


class QSpinRollbackStepStatus(str, Enum):
    PENDING = "pending"
    DRY_RUN_PASSED = "dry_run_passed"
    MISSING_EVIDENCE = "missing_evidence"
    BLOCKED = "blocked"


class QSpinRollbackEvidenceKind(str, Enum):
    FEATURE_FLAGS_DISABLED = "feature_flags_disabled"
    KILL_SWITCH_TRIPPED = "kill_switch_tripped"
    PREVIOUS_BASELINE_RESTORABLE = "previous_baseline_restorable"
    AUDIT_LOG_PRESERVED = "audit_log_preserved"
    NO_STATE_MUTATION = "no_state_mutation"


@dataclass(frozen=True)
class QSpinRollbackStep:
    step_id: str
    title: str
    required_evidence: QSpinRollbackEvidenceKind
    status: QSpinRollbackStepStatus = QSpinRollbackStepStatus.PENDING

    def validate(self) -> "QSpinRollbackStep":
        if not self.step_id or not self.title:
            raise ValueError("rollback step requires step_id and title")
        return self


@dataclass(frozen=True)
class QSpinRollbackEvidenceRecord:
    evidence_id: str
    kind: QSpinRollbackEvidenceKind
    present: bool = True
    secret_free: bool = True
    raw_payload_free: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "QSpinRollbackEvidenceRecord":
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.present:
            raise ValueError("evidence record marked not present")
        if not self.secret_free or not self.raw_payload_free:
            raise ValueError("rollback evidence must be secret-free and raw-payload-free")
        return self


@dataclass(frozen=True)
class QSpinRollbackPlanSnapshot:
    plan_id: str = "qspin_prod1_rollback_plan_snapshot"
    steps: Tuple[QSpinRollbackStep, ...] = ()
    dry_run_only: bool = True
    no_file_deletion: bool = True
    no_runtime_mutation: bool = True

    def validate(self) -> "QSpinRollbackPlanSnapshot":
        if not self.steps:
            raise ValueError("rollback plan requires steps")
        kinds = {step.required_evidence for step in self.steps}
        required = set(QSpinRollbackEvidenceKind)
        missing = required - kinds
        if missing:
            raise ValueError("rollback plan missing evidence kinds: " + ",".join(sorted(k.value for k in missing)))
        if not self.dry_run_only or not self.no_file_deletion or not self.no_runtime_mutation:
            raise ValueError("PROD-1 rollback harness must be dry-run only with no deletion/mutation")
        return self


@dataclass(frozen=True)
class QSpinRollbackDryRunRequest:
    request_id: str
    evidence: Tuple[QSpinRollbackEvidenceRecord, ...]

    def validate(self) -> "QSpinRollbackDryRunRequest":
        if not self.request_id:
            raise ValueError("request_id is required")
        for item in self.evidence:
            item.validate()
        return self


@dataclass(frozen=True)
class QSpinRollbackDryRunResult:
    request_id: str
    passed: bool
    missing_evidence: Tuple[QSpinRollbackEvidenceKind, ...]
    executed_real_rollback: bool = False
    mutated_state: bool = False

    def validate(self) -> "QSpinRollbackDryRunResult":
        if self.executed_real_rollback:
            raise ValueError("PROD-1 rollback dry-run must not execute real rollback")
        if self.mutated_state:
            raise ValueError("PROD-1 rollback dry-run must not mutate state")
        if self.passed and self.missing_evidence:
            raise ValueError("passed rollback dry-run cannot have missing evidence")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "passed": self.passed,
            "missing_evidence": [m.value for m in self.missing_evidence],
            "executed_real_rollback": self.executed_real_rollback,
            "mutated_state": self.mutated_state,
        }


class QSpinRollbackHarness:
    def __init__(self, plan: QSpinRollbackPlanSnapshot):
        self.plan = plan.validate()

    def dry_run(self, request: QSpinRollbackDryRunRequest) -> QSpinRollbackDryRunResult:
        request.validate()
        have = {e.kind for e in request.evidence if e.present}
        required = {step.required_evidence for step in self.plan.steps}
        missing = tuple(sorted(required - have, key=lambda k: k.value))
        return QSpinRollbackDryRunResult(request.request_id, not missing, missing).validate()


def build_default_qspin_rollback_plan_snapshot() -> QSpinRollbackPlanSnapshot:
    steps = (
        QSpinRollbackStep("disable_feature_flags", "Disable all QSPIN runtime feature flags", QSpinRollbackEvidenceKind.FEATURE_FLAGS_DISABLED),
        QSpinRollbackStep("trip_kill_switch", "Trip QSPIN kill switch", QSpinRollbackEvidenceKind.KILL_SWITCH_TRIPPED),
        QSpinRollbackStep("restore_baseline", "Restore previous QSPIN metadata baseline", QSpinRollbackEvidenceKind.PREVIOUS_BASELINE_RESTORABLE),
        QSpinRollbackStep("preserve_audit", "Preserve audit logs and traces", QSpinRollbackEvidenceKind.AUDIT_LOG_PRESERVED),
        QSpinRollbackStep("prove_no_mutation", "Prove dry-run made no state mutation", QSpinRollbackEvidenceKind.NO_STATE_MUTATION),
    )
    return QSpinRollbackPlanSnapshot(steps=steps).validate()


def build_default_qspin_rollback_harness() -> QSpinRollbackHarness:
    return QSpinRollbackHarness(build_default_qspin_rollback_plan_snapshot())
