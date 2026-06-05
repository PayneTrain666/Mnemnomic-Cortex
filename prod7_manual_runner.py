from __future__ import annotations
import json
import importlib.util
import sys
import types
from pathlib import Path

def _load_qspin_module(short_name: str):
    root = Path(__file__).resolve().parent
    wm_dir = root / "mnemonic_cortex" / "working_memory"
    full = f"mnemonic_cortex.working_memory.{short_name}"
    sys.modules.setdefault("mnemonic_cortex", types.ModuleType("mnemonic_cortex"))
    pkg = sys.modules.get("mnemonic_cortex.working_memory")
    if pkg is None:
        pkg = types.ModuleType("mnemonic_cortex.working_memory")
        pkg.__path__ = [str(wm_dir)]
        sys.modules["mnemonic_cortex.working_memory"] = pkg
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, wm_dir / f"{short_name}.py")
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"missing module {short_name} under {wm_dir}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


_probe = _load_qspin_module("qspin_readonly_runtime_probe")
_boundary = _load_qspin_module("qspin_synthetic_real_boundary")
_ci = _load_qspin_module("qspin_ci_gate_enforcement")
_obs_review = _load_qspin_module("qspin_observability_review")
_blockers = _load_qspin_module("qspin_prod_readiness_blockers")
_safety = _load_qspin_module("qspin_runtime_probe_safety_regression")
_obs = _load_qspin_module("qspin_prod7_observability")
_matrix = _load_qspin_module("qspin_prod7_source_matrix")

ReadOnlyRuntimeProbeHarness = _probe.ReadOnlyRuntimeProbeHarness
build_default_readonly_probe_suite = _probe.build_default_readonly_probe_suite
SyntheticToRealBoundaryVerifier = _boundary.SyntheticToRealBoundaryVerifier
build_default_boundary_surface_records = _boundary.build_default_boundary_surface_records
CIGateEnforcer = _ci.CIGateEnforcer
build_default_ci_gate_checks = _ci.build_default_ci_gate_checks
ObservabilityReviewEngine = _obs_review.ObservabilityReviewEngine
ObservabilityReviewRequest = _obs_review.ObservabilityReviewRequest
build_default_observability_signal_records = _obs_review.build_default_observability_signal_records
build_default_readiness_blocker_register = _blockers.build_default_readiness_blocker_register
ProbeSafetyRegressionRunner = _safety.ProbeSafetyRegressionRunner
build_default_probe_safety_cases = _safety.build_default_probe_safety_cases
build_default_prod7_observability_collector = _obs.build_default_prod7_observability_collector
Prod7MetricEvent = _obs.Prod7MetricEvent
Prod7MetricName = _obs.Prod7MetricName
build_prod7_source_matrix = _matrix.build_prod7_source_matrix
export_prod7_source_matrix_json = _matrix.export_prod7_source_matrix_json
export_prod7_source_matrix_markdown = _matrix.export_prod7_source_matrix_markdown

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
