#!/usr/bin/env python3
"""Manual local-only runner for QSPIN-PROD-5 synthetic bridge checks.

Runs with: python3 -S prod5_manual_runner.py
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemonic_cortex.working_memory.qspin_shadow_runtime_harness import ShadowRuntimeHarness
from mnemonic_cortex.working_memory.qspin_synthetic_canary_bridge import run_canaries
from mnemonic_cortex.working_memory.qspin_safety_regression_ci import run_safety_regressions, safety_summary_to_junit_xml
from mnemonic_cortex.working_memory.qspin_source_consideration_matrix import build_source_matrix, matrix_to_json, matrix_to_markdown, consideration_coverage_audit
from mnemonic_cortex.working_memory.qspin_active_remediation_register import build_remediation_register, register_to_json, register_to_markdown


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    outdir = ROOT / "prod5_outputs"
    outdir.mkdir(exist_ok=True)

    records = build_source_matrix()
    source_audit = consideration_coverage_audit(records)
    write(outdir / "source_matrix.json", matrix_to_json(records))
    write(outdir / "source_matrix.md", matrix_to_markdown(records))

    harness = ShadowRuntimeHarness()
    canaries = run_canaries(harness)
    canary_payload = [asdict(c) for c in canaries]
    write(outdir / "canary_results.json", json.dumps(canary_payload, indent=2, sort_keys=True, default=str))

    safety = run_safety_regressions()
    write(outdir / "safety_regression_summary.json", json.dumps(safety, indent=2, sort_keys=True, default=str))
    write(outdir / "safety_regression_junit.xml", safety_summary_to_junit_xml(safety))

    remediations = build_remediation_register(canaries, safety, source_audit)
    write(outdir / "remediation_register.json", register_to_json(remediations))
    write(outdir / "remediation_register.md", register_to_markdown(remediations))

    pass_count = sum(1 for c in canaries if c.status == "PASS") + safety["pass_count"]
    fail_count = sum(1 for c in canaries if c.status == "FAIL") + safety["fail_count"]
    skip_count = len(source_audit.get("skipped_missing", []))
    summary = {
        "stage": "QSPIN-PROD-5-QD6A",
        "status": "PASS" if fail_count == 0 else "FAIL",
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "canary_count": len(canaries),
        "safety_regression_count": len(safety.get("checks", [])),
        "source_audit": source_audit,
        "safety_boundaries": {
            "live_routing": "DISABLED/BLOCKED",
            "real_payload_transfer": "DISABLED/BLOCKED",
            "topology_execution": "DISABLED/BLOCKED",
            "real_shared_slot_writes": "DISABLED/BLOCKED",
            "qh_writes": "DISABLED/BLOCKED",
            "external_memory_writes": "DISABLED/BLOCKED",
            "commit_execution": "DISABLED/BLOCKED",
            "production_activation": "DISABLED/BLOCKED",
        },
        "outputs": {
            "source_matrix_json": str(outdir / "source_matrix.json"),
            "canary_results_json": str(outdir / "canary_results.json"),
            "safety_regression_summary_json": str(outdir / "safety_regression_summary.json"),
            "remediation_register_json": str(outdir / "remediation_register.json"),
        },
    }
    write(outdir / "prod5_summary.json", json.dumps(summary, indent=2, sort_keys=True, default=str))
    md = [
        "# QSPIN-PROD-5-QD6A Manual Runner Summary",
        "",
        f"Status: {summary['status']}",
        f"Pass count: {pass_count}",
        f"Fail count: {fail_count}",
        f"Skip count: {skip_count}",
        "",
        "## Safety Boundaries",
    ]
    for k, v in summary["safety_boundaries"].items():
        md.append(f"- {k}: {v}")
    md.extend(["", "## Source audit", json.dumps(source_audit, indent=2, sort_keys=True)])
    write(outdir / "prod5_summary.md", "\n".join(md) + "\n")

    print(f"QSPIN-PROD-5-QD6A synthetic runner status={summary['status']} pass={pass_count} fail={fail_count} skip={skip_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
