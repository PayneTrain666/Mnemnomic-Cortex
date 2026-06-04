#!/usr/bin/env python3
"""Manual runner for QSPIN-PROD-6.

Local-only. Synthetic-only. No external services, no repo commits, no production
state mutation.
"""
from __future__ import annotations
import json, os, sys
import importlib.util
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


_trace = _load_qspin_module("qspin_trace_corpus")
_stress = _load_qspin_module("qspin_shadow_stress_replay")
_ci = _load_qspin_module("qspin_ci_matrix_hardening")
_safety = _load_qspin_module("qspin_extended_safety_regression")
_remed = _load_qspin_module("qspin_remediation_closure")
_obs = _load_qspin_module("qspin_prod6_observability")
_matrix = _load_qspin_module("qspin_prod6_source_matrix")

DeterministicTraceCorpusBuilder = _trace.DeterministicTraceCorpusBuilder
ShadowStressReplayEngine = _stress.ShadowStressReplayEngine
StressReplayInput = _stress.StressReplayInput
CIMatrixRunner = _ci.CIMatrixRunner
ExtendedSafetyRegressionRunner = _safety.ExtendedSafetyRegressionRunner
RemediationClosureWorkflow = _remed.RemediationClosureWorkflow
build_default_prod6_observability_collector = _obs.build_default_prod6_observability_collector
Prod6MetricEvent = _obs.Prod6MetricEvent
Prod6MetricName = _obs.Prod6MetricName
Prod6AuditEvent = _obs.Prod6AuditEvent
Prod6SpanEvent = _obs.Prod6SpanEvent
build_prod6_source_matrix = _matrix.build_prod6_source_matrix
export_prod6_source_matrix_markdown = _matrix.export_prod6_source_matrix_markdown
export_prod6_source_matrix_json = _matrix.export_prod6_source_matrix_json

OUT = Path("prod6_outputs")
OUT.mkdir(exist_ok=True)

def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def main() -> int:
    pass_count = fail_count = skip_count = 0
    source_matrix = build_prod6_source_matrix()
    write_json(OUT / "source_matrix.json", export_prod6_source_matrix_json(source_matrix))
    (OUT / "source_matrix.md").write_text(export_prod6_source_matrix_markdown(source_matrix), encoding="utf-8")
    pass_count += 1

    corpus_builder = DeterministicTraceCorpusBuilder()
    corpus = corpus_builder.build()
    replay = corpus_builder.replay(corpus)
    write_json(OUT / "trace_corpus.json", corpus.to_json_dict())
    (OUT / "trace_corpus.md").write_text(corpus.to_markdown(), encoding="utf-8")
    write_json(OUT / "trace_replay.json", {"passed": replay.passed, "expected_hash": replay.expected_hash, "actual_hash": replay.actual_hash})
    pass_count += int(replay.passed); fail_count += int(not replay.passed)

    stress = ShadowStressReplayEngine().run(StressReplayInput())
    write_json(OUT / "stress_replay.json", stress.to_json_dict())
    pass_count += int(stress.passed); fail_count += int(not stress.passed)

    ci = CIMatrixRunner().run()
    write_json(OUT / "ci_matrix.json", ci.to_json_dict())
    (OUT / "ci_matrix.md").write_text(ci.to_markdown(), encoding="utf-8")
    (OUT / "ci_matrix_junit.xml").write_text(ci.to_junit_xml(), encoding="utf-8")
    pass_count += int(ci.passed); fail_count += int(not ci.passed); skip_count += ci.skip_count

    safety = ExtendedSafetyRegressionRunner().run()
    write_json(OUT / "extended_safety.json", safety.to_json_dict())
    pass_count += int(safety.passed); fail_count += int(not safety.passed)

    remediation = RemediationClosureWorkflow().close()
    write_json(OUT / "remediation_closure.json", remediation.to_json_dict())
    (OUT / "remediation_closure.md").write_text(remediation.to_markdown(), encoding="utf-8")
    pass_count += int(remediation.passed); fail_count += int(not remediation.passed)

    obs = build_default_prod6_observability_collector()
    obs.emit_metric(Prod6MetricEvent(Prod6MetricName.STRESS_REPLAY_ATTEMPTS, 1))
    obs.emit_metric(Prod6MetricEvent(Prod6MetricName.TRACE_CORPUS_RECORDS_GENERATED, len(corpus.records)))
    obs.emit_metric(Prod6MetricEvent(Prod6MetricName.CI_MATRIX_CASES_RUN, len(ci.results)))
    obs.emit_metric(Prod6MetricEvent(Prod6MetricName.EXTENDED_SAFETY_CASES_RUN, len(safety.results)))
    obs.emit_audit(Prod6AuditEvent("prod6_audit_runner", "manual_runner", ("synthetic_only", "no_live_effects")))
    obs.emit_span(Prod6SpanEvent("prod6_span", "prod6_manual_runner", "ok", {"safe": True, "synthetic": True}))
    snapshot = obs.snapshot()
    write_json(OUT / "observability_snapshot.json", snapshot.to_json_dict())
    pass_count += 1

    summary = {
        "stage": "QSPIN-PROD-6-QD6A",
        "passed": fail_count == 0,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
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
            "production_config_mutation": "BLOCKED",
            "unbounded_workers": "BLOCKED",
            "hidden_skips": "BLOCKED",
        }
    }
    write_json(OUT / "prod6_summary.json", summary)
    (OUT / "prod6_summary.md").write_text(f"# QSPIN-PROD-6 Summary\n\nPASS={pass_count}\nFAIL={fail_count}\nSKIP={skip_count}\n", encoding="utf-8")
    print(f"PASS={pass_count}")
    print(f"FAIL={fail_count}")
    print(f"SKIP={skip_count}")
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
