from __future__ import annotations
import json
from pathlib import Path
from mnemonic_cortex.working_memory.qspin_readonly_runtime_probe import ReadOnlyRuntimeProbeHarness, build_default_readonly_probe_suite
from mnemonic_cortex.working_memory.qspin_synthetic_real_boundary import SyntheticToRealBoundaryVerifier, build_default_boundary_surface_records
from mnemonic_cortex.working_memory.qspin_ci_gate_enforcement import CIGateEnforcer, build_default_ci_gate_checks
from mnemonic_cortex.working_memory.qspin_observability_review import ObservabilityReviewEngine, ObservabilityReviewRequest, build_default_observability_signal_records
from mnemonic_cortex.working_memory.qspin_prod_readiness_blockers import build_default_readiness_blocker_register
from mnemonic_cortex.working_memory.qspin_runtime_probe_safety_regression import ProbeSafetyRegressionRunner, build_default_probe_safety_cases
from mnemonic_cortex.working_memory.qspin_prod7_observability import build_default_prod7_observability_collector, Prod7MetricEvent, Prod7MetricName
from mnemonic_cortex.working_memory.qspin_prod7_source_matrix import build_prod7_source_matrix, export_prod7_source_matrix_json, export_prod7_source_matrix_markdown

out = Path('prod7_outputs')
out.mkdir(exist_ok=True)

matrix = build_prod7_source_matrix()
(out/'source_matrix.json').write_text(export_prod7_source_matrix_json(matrix), encoding='utf-8')
(out/'source_matrix.md').write_text(export_prod7_source_matrix_markdown(matrix), encoding='utf-8')

probe = ReadOnlyRuntimeProbeHarness(root='.')
probe_result = probe.run_suite(build_default_readonly_probe_suite())

boundary = SyntheticToRealBoundaryVerifier().verify_suite(build_default_boundary_surface_records())
ci = CIGateEnforcer().run(build_default_ci_gate_checks())
obs_result = ObservabilityReviewEngine().review(ObservabilityReviewRequest('obs_review', build_default_observability_signal_records()))
blockers = build_default_readiness_blocker_register().report()
safety = ProbeSafetyRegressionRunner().run(build_default_probe_safety_cases())
collector = build_default_prod7_observability_collector()
collector.emit_metric(Prod7MetricEvent(Prod7MetricName.READONLY_PROBE_ATTEMPTS, len(probe_result.results)))
collector.emit_metric(Prod7MetricEvent(Prod7MetricName.BOUNDARY_VERIFICATION_ATTEMPTS, len(boundary.results)))
collector.emit_metric(Prod7MetricEvent(Prod7MetricName.CI_GATE_CHECKS_RUN, len(ci.results)))
collector.emit_metric(Prod7MetricEvent(Prod7MetricName.PROBE_SAFETY_CASES_RUN, len(safety.results)))
snapshot = collector.snapshot()

failed = probe_result.failed + boundary.failed + ci.failed + safety.failed
# Open critical blockers intentionally prevent production-active claim; they do not fail the read-only stage.
summary = {
    'stage': 'QSPIN-PROD-7-QD6A',
    'production_active': False,
    'live_routing': 'BLOCKED',
    'real_payload_transfer': 'BLOCKED',
    'real_writes': 'BLOCKED',
    'commits': 'BLOCKED',
    'probe': probe_result.to_dict(),
    'boundary': boundary.to_dict(),
    'ci': ci.to_dict(),
    'observability_review': obs_result.to_dict(),
    'readiness_blockers': blockers.to_dict(),
    'safety': safety.to_dict(),
    'observability_snapshot': snapshot.to_dict(),
    'pass': probe_result.passed + boundary.passed + ci.passed + safety.passed + (1 if obs_result.status.value in ('passed','gap_found') else 0),
    'fail': failed,
    'skip': probe_result.skipped + boundary.skipped + ci.skipped,
}
(out/'prod7_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
(out/'prod7_summary.md').write_text(f"# PROD-7 Summary\n\nPASS={summary['pass']} FAIL={summary['fail']} SKIP={summary['skip']}\n\nProduction active: false\n", encoding='utf-8')
(out/'prod7_junit.xml').write_text(ci.to_junit_xml(), encoding='utf-8')
print(f"PASS={summary['pass']}")
print(f"FAIL={summary['fail']}")
print(f"SKIP={summary['skip']}")
raise SystemExit(1 if failed else 0)
