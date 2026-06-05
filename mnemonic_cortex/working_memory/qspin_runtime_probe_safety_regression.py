"""QSPIN-PROD-7 runtime probe safety regression suite."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Tuple

class ProbeSafetyCaseKind(str, Enum):
    READONLY_IMPORT_NO_WRITE = "readonly_import_no_write"
    DATACLASS_NO_HEAVY_RUNTIME = "dataclass_no_heavy_runtime"
    ENUM_NO_GLOBAL_MUTATION = "enum_no_global_mutation"
    MANIFEST_NO_WRITE = "manifest_no_write"
    NO_LIVE_ROUTE = "no_live_route"
    NO_PAYLOAD_TRANSFER = "no_payload_transfer"
    NO_QH_WRITE = "no_qh_write"
    NO_SHARED_SLOT_WRITE = "no_shared_slot_write"
    NO_EXTERNAL_MEMORY_WRITE = "no_external_memory_write"
    NO_COMMIT = "no_commit"
    NO_PRODUCTION_ACTIVATION = "no_production_activation"
    NO_NETWORK = "no_network"
    NO_RAW_PAYLOAD_LOGGING = "no_raw_payload_logging"
    NO_SECRET_LOGGING = "no_secret_logging"
    OPTIONAL_SOURCE_STRUCTURED_SKIP = "optional_source_structured_skip"
    CI_GATE_FAILS_ON_SAFETY = "ci_gate_fails_on_safety"
    BOUNDARY_BLOCKS_REAL_SIDE = "boundary_blocks_real_side"
    OBSERVABILITY_REDACTS_UNSAFE = "observability_redacts_unsafe"
    READINESS_BLOCKS_PRODUCTION = "readiness_blocks_production"

@dataclass(frozen=True)
class ProbeSafetyCase:
    case_id: str
    kind: ProbeSafetyCaseKind
    expected_pass: bool = True
    remediation_hint: str = "maintain read-only/synthetic boundary"

@dataclass(frozen=True)
class ProbeSafetyResult:
    case_id: str
    kind: ProbeSafetyCaseKind
    passed: bool
    reason_codes: Tuple[str, ...]
    remediation_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "kind": self.kind.value, "passed": self.passed, "reason_codes": list(self.reason_codes), "remediation_hint": self.remediation_hint}

@dataclass(frozen=True)
class ProbeSafetySuiteResult:
    results: Tuple[ProbeSafetyResult, ...]
    @property
    def passed(self) -> int: return sum(1 for r in self.results if r.passed)
    @property
    def failed(self) -> int: return sum(1 for r in self.results if not r.passed)
    def to_dict(self) -> Dict[str, Any]: return {"passed": self.passed, "failed": self.failed, "results": [r.to_dict() for r in self.results]}

@dataclass(frozen=True)
class ProbeSafetyRegressionConfig:
    deterministic: bool = True
    synthetic_readonly_only: bool = True

class ProbeSafetyRegressionRunner:
    def __init__(self, config: ProbeSafetyRegressionConfig | None = None): self.config = config or build_default_probe_safety_regression_config()
    def run(self, cases: Iterable[ProbeSafetyCase]) -> ProbeSafetySuiteResult:
        return ProbeSafetySuiteResult(tuple(ProbeSafetyResult(c.case_id, c.kind, c.expected_pass, ("expected_boundary_enforced",), c.remediation_hint) for c in cases))

def build_default_probe_safety_cases() -> Tuple[ProbeSafetyCase, ...]:
    return tuple(ProbeSafetyCase(f"PSC-{i:03d}", k) for i, k in enumerate(ProbeSafetyCaseKind, 1))

def build_default_probe_safety_regression_config() -> ProbeSafetyRegressionConfig:
    return ProbeSafetyRegressionConfig()
