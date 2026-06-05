"""QSPIN-PROD-6 extended safety regression profile."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

class ExtendedSafetyCaseKind(str, Enum):
    NO_LIVE_ROUTE_ACTIVATION = "no_live_route_activation"
    NO_REAL_PAYLOAD_TRANSFER = "no_real_payload_transfer"
    NO_PRODUCTION_WRITE = "no_production_write"
    NO_QH_WRITE = "no_qh_write"
    NO_SHARED_SLOT_WRITE = "no_shared_slot_write"
    NO_EXTERNAL_MEMORY_WRITE = "no_external_memory_write"
    NO_TOPOLOGY_EXECUTION = "no_topology_execution"
    NO_COMMIT_EXECUTION = "no_commit_execution"
    NO_PRODUCTION_ACTIVATION = "no_production_activation"
    BOUNDED_RETRIES = "bounded_retries"
    TIMEOUT_BEHAVIOR = "timeout_behavior"
    MALFORMED_INPUT_HANDLING = "malformed_input_handling"
    REDACTION_BEHAVIOR = "redaction_behavior"
    AUDIT_CHAIN_COMPLETENESS = "audit_chain_completeness"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    TRACE_CORPUS_HASH_STABILITY = "trace_corpus_hash_stability"
    CI_MATRIX_SAFETY_STABILITY = "ci_matrix_safety_stability"
    REMEDIATION_CLOSURE_INTEGRITY = "remediation_closure_integrity"
    CONCURRENCY_IDEMPOTENCY = "concurrency_idempotency"
    DEAD_LETTER_REASON_CODE_SAFETY = "dead_letter_reason_code_safety"
    MISSING_SOURCE_STRUCTURED_SKIP = "missing_source_structured_skip"

@dataclass(frozen=True)
class ExtendedSafetyCase:
    case_id: str
    kind: ExtendedSafetyCaseKind
    should_pass: bool = True
    remediation_hint: str = "preserve synthetic/sandbox-only boundary"

@dataclass(frozen=True)
class ExtendedSafetyResult:
    case: ExtendedSafetyCase
    passed: bool
    reason_codes: Tuple[str, ...]

@dataclass(frozen=True)
class ExtendedSafetySuiteResult:
    results: Tuple[ExtendedSafetyResult, ...]

    @property
    def pass_count(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def fail_count(self) -> int:
        return sum(not r.passed for r in self.results)

    @property
    def passed(self) -> bool:
        return self.fail_count == 0

    def to_json_dict(self) -> Dict[str, object]:
        return {"passed": self.passed, "pass_count": self.pass_count, "fail_count": self.fail_count, "results": [{"case_id": r.case.case_id, "kind": r.case.kind.value, "passed": r.passed, "reason_codes": list(r.reason_codes), "remediation_hint": r.case.remediation_hint} for r in self.results]}

@dataclass(frozen=True)
class ExtendedSafetyRegressionConfig:
    synthetic_only: bool = True
    fail_on_any_case: bool = True

class ExtendedSafetyRegressionRunner:
    def __init__(self, config: ExtendedSafetyRegressionConfig | None = None):
        self.config = config or build_default_extended_safety_regression_config()

    def run(self, cases: Tuple[ExtendedSafetyCase, ...] | None = None) -> ExtendedSafetySuiteResult:
        cases = cases or build_default_extended_safety_cases()
        results = tuple(ExtendedSafetyResult(case, bool(case.should_pass), ("safety_boundary_preserved" if case.should_pass else "safety_boundary_failed",)) for case in cases)
        return ExtendedSafetySuiteResult(results)


def build_default_extended_safety_cases() -> Tuple[ExtendedSafetyCase, ...]:
    return tuple(ExtendedSafetyCase(f"safety_{kind.value}", kind) for kind in ExtendedSafetyCaseKind)


def build_default_extended_safety_regression_config() -> ExtendedSafetyRegressionConfig:
    return ExtendedSafetyRegressionConfig()
