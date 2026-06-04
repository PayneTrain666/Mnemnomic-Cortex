"""Production-readiness gate for HGM-9.

The gate scores readiness but never enables production execution. It exists to
separate evaluation evidence from future explicit production activation.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .hgm8_result import HGM8PipelineEvaluationResult
from .hgm9_result import HGM9ReadinessOptions, ProductionReadinessGate, QDTRuntimeEvaluationResult, SlotLatticeReplayBenchmarkResult, hgm9_stable_hash, trace_hgm9
from .qdt_runtime_evaluation import coerce_hgm9_options, evaluate_qdt_runtime_integration
from .slot_lattice_replay_benchmark import benchmark_slot_lattice_replay
from .validation import ValidationResult


def _extract_hgm8(value: Any):
    if isinstance(value, HGM8PipelineEvaluationResult):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _extract_hgm8(item)
            if found is not None:
                return found
    return None


def score_production_readiness(
    records_or_result: Any = None,
    qdt_evaluation: Optional[QDTRuntimeEvaluationResult] = None,
    slot_replay: Optional[SlotLatticeReplayBenchmarkResult] = None,
    config=None,
    options: Optional[HGM9ReadinessOptions | Mapping[str, Any]] = None,
) -> ProductionReadinessGate:
    """Score production readiness without enabling production execution."""

    opts = coerce_hgm9_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    qdt_eval = qdt_evaluation or evaluate_qdt_runtime_integration(records_or_result, config=config, options=opts)
    slot = slot_replay or benchmark_slot_lattice_replay(records_or_result, config=config, options=opts)
    validation.merge(qdt_eval.validation).merge(slot.validation)

    hgm8 = _extract_hgm8(records_or_result)
    pipeline_score = hgm8.benchmark_result.aggregate_score if hgm8 is not None else 0.35
    if hgm8 is None:
        validation.warning("hgm9_readiness.missing_hgm8", "HGM-8 pipeline result missing; conservative score applied", "hgm8")
    replay_score = slot.aggregate_score if slot.records else 0.35
    qdt_score = qdt_eval.integration_score
    score = max(0.0, min(1.0, (qdt_score * 0.40) + (replay_score * 0.30) + (pipeline_score * 0.30)))

    blockers: List[str] = []
    warnings: List[str] = []
    if opts.require_adapter_available and not qdt_eval.adapter_available:
        blockers.append("adapter unavailable")
    elif not qdt_eval.adapter_available:
        warnings.append("adapter unavailable; evaluation-only readiness")
    if opts.require_pipeline_passed and (hgm8 is None or not hgm8.benchmark_result.passed):
        blockers.append("pipeline benchmark not passed")
    if not slot.replay_safe:
        warnings.append("slot-lattice replay is not fully safe or no hooks were supplied")
    if score < opts.readiness_threshold:
        blockers.append(f"readiness score below threshold {opts.readiness_threshold:.2f}")
    if not opts.production_enable_allowed:
        blockers.append("production enablement is disabled by HGM-9 gate")

    ready = not blockers and score >= opts.readiness_threshold
    production_enabled = False
    trace = trace_hgm9("production_readiness_gate.score_production_readiness", validation, {"score": score, "ready": ready, "blockers": tuple(blockers), "secret_token": "must_redact"})
    traces.append(trace)
    return ProductionReadinessGate(
        gate_id=f"hgm9_prod_gate_{hgm9_stable_hash(score, tuple(blockers), tuple(warnings), opts.readiness_threshold)}",
        score=score,
        ready=bool(ready),
        production_enabled=production_enabled,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        trace_id=trace.trace_id,
        metadata={
            "evaluation_first": True,
            "live_qdt_write": False,
            "production_execution_enabled": False,
            "threshold": opts.readiness_threshold,
            "qdt_score": qdt_score,
            "slot_replay_score": replay_score,
            "pipeline_score": pipeline_score,
        },
    )
