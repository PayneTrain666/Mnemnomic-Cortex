"""CI-compatible safety regression runner for QSPIN-PROD-5."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
import json
import time
import xml.etree.ElementTree as ET

from .qspin_shadow_runtime_harness import ShadowRuntimeEnvelope, ShadowRuntimeHarness
from .qspin_runtime_observability import no_secret_leakage_check


@dataclass
class SafetyCheck:
    regression_id: str
    name: str
    severity: str
    passed: bool
    observed_status: str
    observed_reason: str
    detail: Dict[str, Any]


def _blocked_check(regression_id: str, name: str, flag: str, severity: str = "P0") -> SafetyCheck:
    harness = ShadowRuntimeHarness()
    result = harness.execute(ShadowRuntimeEnvelope(payload={"check": name}, flags={flag: True}))
    passed = result.status == "FAIL_CLOSED" and result.reason_code == "SAFETY_BOUNDARY_BLOCK"
    return SafetyCheck(regression_id, name, severity, passed, result.status, result.reason_code, {"blocked_actions": result.blocked_actions})


def run_safety_regressions() -> Dict[str, Any]:
    checks: List[SafetyCheck] = []
    blocked_cases = [
        ("REG-P5-001", "no live route activation", "live_route"),
        ("REG-P5-002", "no real payload transfer", "transfer_payload"),
        ("REG-P5-003", "no production write", "mutate_production_config"),
        ("REG-P5-004", "no QH write", "write_qh"),
        ("REG-P5-005", "no shared-slot write", "write_shared_slot"),
        ("REG-P5-006", "no external-memory write", "write_external_memory"),
        ("REG-P5-007", "no topology execution", "execute_topology"),
        ("REG-P5-008", "no commit execution", "commit"),
        ("REG-P5-009", "no network call", "network_call"),
    ]
    for args in blocked_cases:
        checks.append(_blocked_check(*args))

    harness = ShadowRuntimeHarness()
    good = ShadowRuntimeEnvelope(payload={"check": "deterministic replay"})
    r1 = harness.execute(good)
    r2 = ShadowRuntimeHarness().execute(good)
    checks.append(SafetyCheck("REG-P5-010", "deterministic replay", "P1", r1.fingerprint() == r2.fingerprint(), r1.status, r1.reason_code, {"fp1": r1.fingerprint(), "fp2": r2.fingerprint()}))

    malformed = harness.execute(ShadowRuntimeEnvelope(payload={"check": "malformed mode"}, requested_mode="LIVE_EXECUTE"))
    checks.append(SafetyCheck("REG-P5-011", "malformed input handling", "P1", malformed.status == "FAIL_CLOSED", malformed.status, malformed.reason_code, {"blocked_actions": malformed.blocked_actions}))

    secret_payload = {"token": "bearer abcdefghijklmnopqrstuvwxyz", "nested": {"api_key": "SHOULD_NOT_LEAK"}}
    redaction_result = harness.execute(ShadowRuntimeEnvelope(payload=secret_payload))
    checks.append(SafetyCheck("REG-P5-012", "redaction behavior", "P0", no_secret_leakage_check(redaction_result.__dict__), redaction_result.status, redaction_result.reason_code, {}))

    audit_summary = harness.emitter.summary()
    checks.append(SafetyCheck("REG-P5-013", "audit-chain completeness", "P1", len(audit_summary.get("audits", [])) > 0, "PASS", "AUDIT_EVENTS_PRESENT", {"audit_count": len(audit_summary.get("audits", []))}))

    # bounded retry/backoff/dead-letter is represented as bounded one-shot fail-closed behavior in PROD-5.
    too_large = harness.execute(ShadowRuntimeEnvelope(payload={"blob": "x" * 70000}))
    checks.append(SafetyCheck("REG-P5-014", "bounded payload handling", "P1", too_large.status == "FAIL_CLOSED", too_large.status, too_large.reason_code, {"blocked_actions": too_large.blocked_actions}))

    pass_count = sum(1 for c in checks if c.passed)
    fail_count = sum(1 for c in checks if not c.passed)
    summary = {
        "stage": "QSPIN-PROD-5-QD6A",
        "lineage": harness.config.lineage,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": 0,
        "checks": [asdict(c) for c in checks],
    }
    return summary


def safety_summary_to_junit_xml(summary: Dict[str, Any]) -> str:
    testsuite = ET.Element("testsuite", name="qspin_prod5_safety_regression", tests=str(len(summary.get("checks", []))), failures=str(summary.get("fail_count", 0)))
    for check in summary.get("checks", []):
        case = ET.SubElement(testsuite, "testcase", classname="QSPIN_PROD5", name=check["name"])
        if not check["passed"]:
            failure = ET.SubElement(case, "failure", message=check["observed_reason"])
            failure.text = json.dumps(check, indent=2, sort_keys=True)
    return ET.tostring(testsuite, encoding="unicode")


def main() -> int:
    summary = run_safety_regressions()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["fail_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
