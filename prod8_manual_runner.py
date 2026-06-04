#!/usr/bin/env python3
"""Manual runner for QSPIN-PROD-8.

Local-only, read-only/pre-activation. Produces JSON/markdown/XML outputs under
./prod8_outputs and never performs production activation.
"""
from __future__ import annotations
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "mnemonic_cortex" / "working_memory"))

from qspin_final_readiness_review import FinalPreActivationReadinessReviewer, FinalReadinessReviewRequest, build_default_final_readiness_evidence
from qspin_prod_blocker_burndown import ProductionBlockerBurnDownPlanner, build_default_blocker_burndown_plan
from qspin_readonly_probe_report import ReadOnlyRuntimeProbeReportGenerator
from qspin_ci_gate_baseline_freeze import CIGateBaselineFreezer, CIBaselineFreezeRequest, build_default_ci_baseline_gate_records
from qspin_prod8_release_builder import Prod8ExpandedReleaseBuilder, Prod8ReleaseBuildRequest, build_default_prod8_release_artifacts
from qspin_final_remediation_register import FinalRemediationReport, build_default_final_remediation_register
from qspin_prod8_observability import build_default_prod8_observability_collector, Prod8MetricEvent, Prod8MetricName, Prod8AuditEvent, Prod8EvidenceRecord
from qspin_prod8_source_matrix import build_prod8_source_matrix, export_prod8_source_matrix_json, export_prod8_source_matrix_markdown


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    out = Path("prod8_outputs")
    out.mkdir(exist_ok=True)
    pass_count = 0
    fail_count = 0
    skip_count = 0

    matrix = build_prod8_source_matrix()
    write(out/"prod8_source_matrix.json", export_prod8_source_matrix_json(matrix))
    write(out/"prod8_source_matrix.md", export_prod8_source_matrix_markdown(matrix))
    pass_count += 1

    reviewer = FinalPreActivationReadinessReviewer()
    readiness = reviewer.review(FinalReadinessReviewRequest("prod8_readiness", build_default_final_readiness_evidence(), critical_blockers_open=2))
    write(out/"final_readiness_review.json", readiness.to_json())
    write(out/"final_readiness_review.md", readiness.to_markdown())
    # Expected: not production-ready and hold required.
    pass_count += 1 if (not readiness.production_ready and readiness.hold_required) else 0
    fail_count += 0 if (not readiness.production_ready and readiness.hold_required) else 1

    burndown = ProductionBlockerBurnDownPlanner().plan(build_default_blocker_burndown_plan())
    write(out/"blocker_burndown.json", burndown.to_json())
    write(out/"blocker_burndown.md", burndown.to_markdown())
    pass_count += 1 if burndown.p0_blockers else 0
    fail_count += 0 if burndown.p0_blockers else 1

    probe = ReadOnlyRuntimeProbeReportGenerator().generate()
    write(out/"readonly_probe_report.json", probe.to_json())
    write(out/"readonly_probe_report.md", probe.to_markdown())
    pass_count += 1

    freeze = CIGateBaselineFreezer().freeze(CIBaselineFreezeRequest("prod8_ci_freeze", build_default_ci_baseline_gate_records()))
    write(out/"ci_baseline_freeze.json", freeze.to_json())
    write(out/"ci_baseline_freeze.md", freeze.to_markdown())
    pass_count += 1 if freeze.status.value == "frozen" else 0
    fail_count += 0 if freeze.status.value == "frozen" else 1

    remediation = FinalRemediationReport.from_register(build_default_final_remediation_register())
    write(out/"final_remediation_register.json", remediation.to_json())
    write(out/"final_remediation_register.md", remediation.to_markdown())
    pass_count += 1 if remediation.hold_required else 0
    fail_count += 0 if remediation.hold_required else 1

    release_result = Prod8ExpandedReleaseBuilder().build(Prod8ReleaseBuildRequest("prod8_release", build_default_prod8_release_artifacts()))
    write(out/"prod8_release_build_result.json", release_result.to_json())
    pass_count += 1 if release_result.status.value == "ready_to_package" else 0
    fail_count += 0 if release_result.status.value == "ready_to_package" else 1

    collector = build_default_prod8_observability_collector()
    collector.emit_metric(Prod8MetricEvent(Prod8MetricName.FINAL_READINESS_DOMAINS_REVIEWED, len(readiness.domain_results)))
    collector.emit_metric(Prod8MetricEvent(Prod8MetricName.PRODUCTION_BLOCKERS_P0, len(burndown.p0_blockers)))
    collector.emit_metric(Prod8MetricEvent(Prod8MetricName.CI_BASELINE_GATES_FROZEN, len(freeze.gates)))
    collector.emit_audit(Prod8AuditEvent("audit_prod8_hold", "final_hold_required", ("no_activation",)))
    collector.emit_evidence(Prod8EvidenceRecord("evidence_final_hold", "final_hold", {"hold_required": True}))
    snapshot = collector.snapshot()
    write(out/"prod8_observability_snapshot.json", snapshot.to_json())
    pass_count += 1

    summary = {
        "stage": "QSPIN-PROD-8-QD6A",
        "status": "SHIP_FINAL_PRE_ACTIVATION_HOLD" if fail_count == 0 else "HOLD",
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "production_active": False,
        "final_hold_required": True,
        "safety_boundaries": {
            "live_routing": "BLOCKED",
            "real_payload_transfer": "BLOCKED",
            "topology_execution": "BLOCKED",
            "real_shared_slot_writes": "BLOCKED",
            "qh_writes": "BLOCKED",
            "external_memory_writes": "BLOCKED",
            "commit_execution": "BLOCKED",
            "production_activation": "BLOCKED",
            "raw_payload_logging": "BLOCKED",
            "secret_logging": "BLOCKED",
            "network_calls": "BLOCKED",
            "repo_commits": "BLOCKED",
            "production_ready_claim": "BLOCKED",
        },
    }
    write(out/"prod8_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    md = "# PROD-8 Summary\n\n" + "\n".join(f"- {k}: `{v}`" for k,v in summary.items() if k not in {"safety_boundaries"}) + "\n\n## Safety Boundaries\n" + "\n".join(f"- {k}: `{v}`" for k,v in summary["safety_boundaries"].items()) + "\n"
    write(out/"prod8_summary.md", md)
    xml = f'<testsuite name="qspin_prod8" tests="{pass_count+fail_count+skip_count}" failures="{fail_count}" skipped="{skip_count}"></testsuite>\n'
    write(out/"prod8_junit.xml", xml)
    print(f"PASS={pass_count}\nFAIL={fail_count}\nSKIP={skip_count}")
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
