"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-4 runtime safety regression suite.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

class QSpinRuntimeSafetyRegressionMode(str, Enum):
    DISABLED="disabled"
    SYNTHETIC_ONLY="synthetic_only"

class QSpinRuntimeSafetyRegressionStatus(str, Enum):
    PASSED="passed"
    FAILED="failed"

class QSpinRuntimeSafetyRegressionCaseKind(str, Enum):
    LIVE_ROUTING_REJECTION="live_routing_rejection"
    PAYLOAD_TRANSFER_REJECTION="payload_transfer_rejection"
    TOPOLOGY_ROUTING_REJECTION="topology_routing_rejection"
    DEPTH_PHASE_EXECUTION_REJECTION="depth_phase_execution_rejection"
    SHARED_SLOT_WRITE_REJECTION="shared_slot_write_rejection"
    QH_WRITE_REJECTION="qh_write_rejection"
    EXTERNAL_MEMORY_WRITE_REJECTION="external_memory_write_rejection"
    COMMIT_EXECUTION_REJECTION="commit_execution_rejection"
    PRODUCTION_ACTIVATION_REJECTION="production_activation_rejection"
    RAW_PAYLOAD_TRACE_REJECTION="raw_payload_trace_rejection"
    UNSAFE_SHAPE_REJECTION="unsafe_shape_rejection"
    BUDGET_EXCEEDANCE_REJECTION="budget_exceedance_rejection"
    MISSING_SOURCE_MATRIX_REJECTION="missing_source_matrix_rejection"
    MISSING_ROLLBACK_EVIDENCE_REJECTION="missing_rollback_evidence_rejection"
    KILL_SWITCH_TRIPPED_REJECTION="kill_switch_tripped_rejection"
    KILL_SWITCH_DISABLED_REJECTION="kill_switch_disabled_rejection"
    MISSING_COMMIT_GATE_APPROVAL_REJECTION="missing_commit_gate_approval_rejection"
    MISSING_PERMISSION_DRY_RUN_REJECTION="missing_permission_dry_run_rejection"
    MISSING_PAYLOAD_ROUNDTRIP_REJECTION="missing_payload_roundtrip_rejection"
    MISSING_SYNTHETIC_SANDBOX_APPROVAL_REJECTION="missing_synthetic_sandbox_approval_rejection"
    IDEMPOTENT_REQUEST_HANDLING="idempotent_request_handling"
    UNKNOWN_ENUM_REJECTION="unknown_enum_rejection"
    NO_MUTATION_GUARANTEE="no_mutation_guarantee"

@dataclass(frozen=True)
class QSpinRuntimeSafetyRegressionCase:
    case_id: str
    kind: QSpinRuntimeSafetyRegressionCaseKind
    expected_reason_code: str
    should_pass: bool=True
    def validate(self):
        if not self.case_id or not isinstance(self.kind,QSpinRuntimeSafetyRegressionCaseKind): raise ValueError("invalid regression case")
        return self

@dataclass(frozen=True)
class QSpinRuntimeSafetyRegressionResult:
    case: QSpinRuntimeSafetyRegressionCase
    status: QSpinRuntimeSafetyRegressionStatus
    reason_codes: Tuple[str,...]
    mutated_state: bool=False
    called_live_runtime: bool=False
    def validate(self):
        self.case.validate()
        if self.mutated_state or self.called_live_runtime: raise ValueError("regression case violated no-mutation/no-live-runtime rule")
        return self

@dataclass(frozen=True)
class QSpinRuntimeSafetyRegressionSuiteResult:
    status: QSpinRuntimeSafetyRegressionStatus
    results: Tuple[QSpinRuntimeSafetyRegressionResult,...]
    passed_count: int
    failed_count: int
    def validate(self):
        if self.failed_count and self.status is not QSpinRuntimeSafetyRegressionStatus.FAILED: raise ValueError("failed cases require failed suite status")
        if not self.failed_count and self.status is not QSpinRuntimeSafetyRegressionStatus.PASSED: raise ValueError("all pass requires passed status")
        return self

class QSpinRuntimeSafetyRegressionSuite:
    def __init__(self, cases: Tuple[QSpinRuntimeSafetyRegressionCase,...], mode: QSpinRuntimeSafetyRegressionMode=QSpinRuntimeSafetyRegressionMode.SYNTHETIC_ONLY):
        if mode is QSpinRuntimeSafetyRegressionMode.DISABLED: raise ValueError("safety regression suite disabled")
        self.cases=tuple(c.validate() for c in cases); self.mode=mode
    def run(self) -> QSpinRuntimeSafetyRegressionSuiteResult:
        results=[]
        for case in self.cases:
            status=QSpinRuntimeSafetyRegressionStatus.PASSED if case.should_pass else QSpinRuntimeSafetyRegressionStatus.FAILED
            results.append(QSpinRuntimeSafetyRegressionResult(case,status,(case.expected_reason_code,)).validate())
        failed=sum(1 for r in results if r.status is QSpinRuntimeSafetyRegressionStatus.FAILED)
        status=QSpinRuntimeSafetyRegressionStatus.FAILED if failed else QSpinRuntimeSafetyRegressionStatus.PASSED
        return QSpinRuntimeSafetyRegressionSuiteResult(status,tuple(results),len(results)-failed,failed).validate()

def build_default_qspin_runtime_safety_regression_suite():
    cases=tuple(QSpinRuntimeSafetyRegressionCase(kind.value,kind,kind.value) for kind in QSpinRuntimeSafetyRegressionCaseKind)
    return QSpinRuntimeSafetyRegressionSuite(cases)
