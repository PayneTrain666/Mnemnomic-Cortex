"""Active remediation register for QSPIN-PROD-5."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List
import json


@dataclass
class RemediationItem:
    remediation_id: str
    severity: str
    status: str
    owner: str
    source_stage: str
    affected_module: str
    failure_mode: str
    recommended_fix: str
    safety_implication: str
    test_coverage: str
    acceptance_criteria: str
    prod6_carry_forward: bool


def build_remediation_register(canary_results: Iterable[Any], safety_summary: Dict[str, Any], source_audit: Dict[str, Any]) -> List[RemediationItem]:
    items: List[RemediationItem] = []
    idx = 1
    for result in canary_results:
        if getattr(result, "status", "PASS") != "PASS":
            items.append(RemediationItem(
                remediation_id=f"PROD5-REM-{idx:03d}", severity="P0", status="open", owner="unassigned",
                source_stage="QSPIN-PROD-5-QD6A", affected_module="qspin_synthetic_canary_bridge.py",
                failure_mode=f"Canary {getattr(result, 'canary_id', 'unknown')} failed: {getattr(result, 'observed_reason_code', '')}",
                recommended_fix=getattr(result, "remediation_hint", "Inspect canary failure."),
                safety_implication="Potential unsafe shadow/live boundary regression.",
                test_coverage="Synthetic canary corpus", acceptance_criteria="Canary returns PASS with expected blocked/pass posture.",
                prod6_carry_forward=True,
            ))
            idx += 1
    for check in safety_summary.get("checks", []):
        if not check.get("passed", False):
            items.append(RemediationItem(
                remediation_id=f"PROD5-REM-{idx:03d}", severity=check.get("severity", "P1"), status="open", owner="unassigned",
                source_stage="QSPIN-PROD-5-QD6A", affected_module="qspin_safety_regression_ci.py",
                failure_mode=f"Safety regression failed: {check.get('name')}",
                recommended_fix="Repair fail-closed validator, regression expectation, or observability audit path.",
                safety_implication="Could permit or mask forbidden behavior.", test_coverage="Safety regression CI",
                acceptance_criteria="Regression passes locally and in CI-compatible runner.", prod6_carry_forward=True,
            ))
            idx += 1
    for skipped in source_audit.get("skipped_missing", []):
        items.append(RemediationItem(
            remediation_id=f"PROD5-REM-{idx:03d}", severity="P2", status="deferred", owner="unassigned",
            source_stage="QSPIN-PROD-5-QD6A", affected_module="qspin_source_consideration_matrix.py",
            failure_mode=f"Source missing: {skipped}", recommended_fix="Provide missing lineage source or mark as intentionally unavailable.",
            safety_implication="Lineage completeness reduced; runtime safety remains fail-closed.", test_coverage="Source consideration matrix",
            acceptance_criteria="Source present or formally waived with documented reason.", prod6_carry_forward=True,
        ))
        idx += 1
    return items


def register_to_json(items: List[RemediationItem]) -> str:
    return json.dumps([asdict(i) for i in items], indent=2, sort_keys=True)


def register_to_markdown(items: List[RemediationItem]) -> str:
    lines = ["# QSPIN-PROD-5-QD6A Active Remediation Register", "", "| ID | Severity | Status | Module | Failure mode | PROD-6 carry-forward |", "|---|---|---|---|---|---:|"]
    if not items:
        lines.append("| none | none | resolved | all | No active remediation items generated. | False |")
    for item in items:
        lines.append(f"| {item.remediation_id} | {item.severity} | {item.status} | {item.affected_module} | {item.failure_mode} | {item.prod6_carry_forward} |")
    return "\n".join(lines) + "\n"
