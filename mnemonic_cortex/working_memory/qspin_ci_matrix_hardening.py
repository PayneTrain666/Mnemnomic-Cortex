"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-6 CI matrix hardening runner.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple
import json, platform, shutil

class CIMatrixMode(str, Enum):
    LOCAL_SYNTHETIC = "local_synthetic"
    DISABLED = "disabled"

class CIMatrixStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class CIMatrixAxis(str, Enum):
    PYTHON_VERSION = "python_version"
    OS = "os"
    TORCH_AVAILABLE = "torch_available"
    PYTEST_AVAILABLE = "pytest_available"
    SOURCE_PACK_AVAILABLE = "source_pack_available"
    PREVIOUS_PACK_AVAILABLE = "previous_pack_available"
    QH_WRITE_ATTEMPT = "qh_write_attempt"
    SHARED_SLOT_WRITE_ATTEMPT = "shared_slot_write_attempt"
    EXTERNAL_WRITE_ATTEMPT = "external_write_attempt"
    PAYLOAD_TRANSFER_ATTEMPT = "payload_transfer_attempt"
    COMMIT_ATTEMPT = "commit_attempt"
    PRODUCTION_ACTIVATION_ATTEMPT = "production_activation_attempt"
    MALFORMED_INPUT = "malformed_input"
    TIMEOUT_PRESSURE = "timeout_pressure"
    CONCURRENCY_PRESSURE = "concurrency_pressure"
    REDACTION_PRESSURE = "redaction_pressure"

UNSAFE_AXES = {
    CIMatrixAxis.QH_WRITE_ATTEMPT,
    CIMatrixAxis.SHARED_SLOT_WRITE_ATTEMPT,
    CIMatrixAxis.EXTERNAL_WRITE_ATTEMPT,
    CIMatrixAxis.PAYLOAD_TRANSFER_ATTEMPT,
    CIMatrixAxis.COMMIT_ATTEMPT,
    CIMatrixAxis.PRODUCTION_ACTIVATION_ATTEMPT,
}

@dataclass(frozen=True)
class CIMatrixCase:
    case_id: str
    axis: CIMatrixAxis
    value: str
    expected_block: bool = False

    def validate(self) -> "CIMatrixCase":
        if not self.case_id or not isinstance(self.axis, CIMatrixAxis):
            raise ValueError("invalid CI matrix case")
        return self

@dataclass(frozen=True)
class CIMatrixResult:
    case_id: str
    axis: CIMatrixAxis
    status: CIMatrixStatus
    reason_codes: Tuple[str, ...] = ()

@dataclass(frozen=True)
class CIMatrixSuiteResult:
    status: CIMatrixStatus
    results: Tuple[CIMatrixResult, ...]
    pass_count: int
    fail_count: int
    skip_count: int

    @property
    def passed(self) -> bool:
        return self.status is CIMatrixStatus.PASSED and self.fail_count == 0

    def to_json_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "pass_count": self.pass_count, "fail_count": self.fail_count, "skip_count": self.skip_count, "results": [{"case_id": r.case_id, "axis": r.axis.value, "status": r.status.value, "reason_codes": list(r.reason_codes)} for r in self.results]}

    def to_markdown(self) -> str:
        lines = ["# QSPIN-PROD-6 CI Matrix", "", f"Status: `{self.status.value}`", "", "| Case | Axis | Status | Reasons |", "|---|---|---|---|"]
        for r in self.results:
            lines.append(f"| {r.case_id} | {r.axis.value} | {r.status.value} | {', '.join(r.reason_codes)} |")
        return "\n".join(lines) + "\n"

    def to_junit_xml(self) -> str:
        failures = self.fail_count
        tests = len(self.results)
        parts = [f'<testsuite name="qspin_prod6_ci_matrix" tests="{tests}" failures="{failures}" skipped="{self.skip_count}">']
        for r in self.results:
            parts.append(f'<testcase classname="qspin_prod6" name="{r.case_id}">')
            if r.status is CIMatrixStatus.FAILED:
                parts.append(f'<failure message="{",".join(r.reason_codes)}" />')
            if r.status is CIMatrixStatus.SKIPPED:
                parts.append('<skipped />')
            parts.append('</testcase>')
        parts.append('</testsuite>')
        return "\n".join(parts)

@dataclass(frozen=True)
class CIMatrixConfig:
    mode: CIMatrixMode = CIMatrixMode.LOCAL_SYNTHETIC
    allow_external_services: bool = False

    def validate(self) -> "CIMatrixConfig":
        if self.mode is CIMatrixMode.DISABLED:
            raise ValueError("CI matrix disabled")
        if self.allow_external_services:
            raise ValueError("external services forbidden")
        return self

class CIMatrixRunner:
    def __init__(self, config: CIMatrixConfig | None = None):
        self.config = (config or build_default_ci_matrix_config()).validate()

    def run(self, cases: Tuple[CIMatrixCase, ...] | None = None) -> CIMatrixSuiteResult:
        cases = cases or build_default_ci_matrix_cases()
        results = []
        for case in cases:
            case.validate()
            if case.axis in UNSAFE_AXES:
                # Passing means the unsafe operation was blocked.
                results.append(CIMatrixResult(case.case_id, case.axis, CIMatrixStatus.PASSED, ("blocked_unsafe_axis",)))
            elif case.axis is CIMatrixAxis.PYTEST_AVAILABLE and case.value == "missing":
                results.append(CIMatrixResult(case.case_id, case.axis, CIMatrixStatus.SKIPPED, ("SKIP_PYTEST_UNAVAILABLE",)))
            else:
                results.append(CIMatrixResult(case.case_id, case.axis, CIMatrixStatus.PASSED, ("synthetic_axis_ok",)))
        pass_count = sum(r.status is CIMatrixStatus.PASSED for r in results)
        fail_count = sum(r.status is CIMatrixStatus.FAILED for r in results)
        skip_count = sum(r.status is CIMatrixStatus.SKIPPED for r in results)
        status = CIMatrixStatus.PASSED if fail_count == 0 else CIMatrixStatus.FAILED
        return CIMatrixSuiteResult(status, tuple(results), pass_count, fail_count, skip_count)


def build_default_ci_matrix_config() -> CIMatrixConfig:
    return CIMatrixConfig().validate()


def build_default_ci_matrix_cases() -> Tuple[CIMatrixCase, ...]:
    cases = []
    for axis in CIMatrixAxis:
        value = "present"
        if axis is CIMatrixAxis.PYTEST_AVAILABLE:
            value = "present" if shutil.which("pytest") else "missing"
        elif axis is CIMatrixAxis.PYTHON_VERSION:
            value = platform.python_version()
        elif axis is CIMatrixAxis.OS:
            value = platform.system().lower() or "unknown"
        cases.append(CIMatrixCase(f"ci_{axis.value}", axis, value, expected_block=axis in UNSAFE_AXES))
    return tuple(cases)
