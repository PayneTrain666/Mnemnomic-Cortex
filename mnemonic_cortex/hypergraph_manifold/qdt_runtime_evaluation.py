"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt runtime evaluation.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
QDT/HGM runtime integration evaluation for HGM-9.

The module inspects and scores contracts only. It never writes into QDT/WM,
never imports heavy runtime modules, and never enables production execution.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from .hgm4_result import BridgeAdapterStatus, TraceSafeMemoryPlan
from .hgm8_result import HGM8PipelineEvaluationResult, PipelineBenchmarkResult, RuntimeEmbeddingTrainerResult, SafeWriteReplayResult
from .hgm9_result import HGM9ReadinessOptions, QDTRuntimeEvaluationResult, QDTRuntimeReadinessMetric, hgm9_stable_hash, trace_hgm9
from .qdt_wm_bridge import detect_qdt_wm_adapter_status
from .validation import ValidationResult


def coerce_hgm9_options(options: Optional[HGM9ReadinessOptions | Mapping[str, Any]]) -> HGM9ReadinessOptions:
    if options is None:
        return HGM9ReadinessOptions()
    if isinstance(options, HGM9ReadinessOptions):
        return options
    return HGM9ReadinessOptions(**dict(options))


def _weighted_average(metrics: Sequence[QDTRuntimeReadinessMetric]) -> float:
    total = sum(m.weight for m in metrics)
    if total <= 0:
        return 0.0
    return sum(m.score * m.weight for m in metrics) / total


def _make_metric(name: str, score: float, weight: float, explanation: str, validation: ValidationResult, traces: List[Any], metadata=None) -> QDTRuntimeReadinessMetric:
    trace = trace_hgm9("qdt_runtime_evaluation.metric", validation, {"metric_name": name, "score": score, "metadata": metadata or {}})
    traces.append(trace)
    return QDTRuntimeReadinessMetric(
        metric_id=f"hgm9_qdt_metric_{hgm9_stable_hash(name, score, weight, explanation)}",
        metric_name=name,
        score=score,
        weight=weight,
        explanation=explanation,
        trace_id=trace.trace_id,
        metadata=dict(metadata or {}),
    )


def _extract_hgm8(records_or_result: Any):
    if isinstance(records_or_result, HGM8PipelineEvaluationResult):
        return records_or_result
    if isinstance(records_or_result, (list, tuple)):
        for item in records_or_result:
            found = _extract_hgm8(item)
            if found is not None:
                return found
    return None


def _extract_plan(records_or_result: Any):
    if isinstance(records_or_result, TraceSafeMemoryPlan):
        return records_or_result
    if isinstance(records_or_result, (list, tuple)):
        for item in records_or_result:
            found = _extract_plan(item)
            if found is not None:
                return found
    return None


def evaluate_qdt_runtime_integration(records_or_result: Any = None, config=None, options: Optional[HGM9ReadinessOptions | Mapping[str, Any]] = None) -> QDTRuntimeEvaluationResult:
    """Evaluate QDT/HGM runtime readiness from HGM-8 and HGM-4 contracts."""

    opts = coerce_hgm9_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    hgm8 = _extract_hgm8(records_or_result)
    plan = _extract_plan(records_or_result)
    adapter: BridgeAdapterStatus = plan.adapter_status if plan is not None else detect_qdt_wm_adapter_status(config=config)
    if not adapter.available:
        validation.warning("hgm9_qdt.adapter_unavailable", "QDT/WM adapter unavailable; readiness remains evaluation-only", "adapter")

    metrics: List[QDTRuntimeReadinessMetric] = []
    adapter_score = 1.0 if adapter.available else 0.35
    metrics.append(_make_metric("adapter_contract_presence", adapter_score, 1.0, "adapter/path presence detected without heavy imports", validation, traces, {"adapter_available": adapter.available}))

    pipeline_score = 0.0
    pipeline_passed = False
    if hgm8 is not None:
        pipeline_score = hgm8.benchmark_result.aggregate_score
        pipeline_passed = hgm8.benchmark_result.passed
    else:
        validation.warning("hgm9_qdt.missing_hgm8", "HGM-8 pipeline result missing; using conservative pipeline score", "hgm8")
        pipeline_score = 0.35
    metrics.append(_make_metric("hgm8_pipeline_score", pipeline_score, 1.25, "HGM-8 end-to-end benchmark score", validation, traces, {"pipeline_passed": pipeline_passed}))

    embedding_score = 0.0
    if hgm8 is not None:
        embeddings = tuple(hgm8.trainer_result.embeddings)
        embedding_score = 1.0 if embeddings else 0.25
    else:
        embedding_score = 0.35
    metrics.append(_make_metric("runtime_embedding_presence", embedding_score, 0.75, "runtime embeddings are available for replay/evaluation", validation, traces))

    replay_score = 0.0
    no_live_write = True
    if hgm8 is not None:
        replay_score = hgm8.replay_result.aggregate_score
        no_live_write = bool(hgm8.replay_result.metadata.get("live_qdt_write") is False)
    else:
        replay_score = 0.35
    metrics.append(_make_metric("safe_write_replay_score", replay_score, 1.0, "safe replay score from HGM-8 logs/results", validation, traces, {"no_live_write": no_live_write}))
    no_write_score = 1.0 if no_live_write else 0.0
    metrics.append(_make_metric("no_live_write_contract", no_write_score, 1.5, "runtime integration evaluation does not perform live writes", validation, traces))

    score = _weighted_average(metrics)
    runtime_ready = score >= opts.readiness_threshold
    if opts.require_adapter_available and not adapter.available:
        runtime_ready = False
    if opts.require_pipeline_passed and not pipeline_passed:
        runtime_ready = False
    if opts.require_no_live_write and not no_live_write:
        runtime_ready = False
    trace = trace_hgm9("qdt_runtime_evaluation.evaluate_qdt_runtime_integration", validation, {"score": score, "runtime_ready": runtime_ready, "secret_token": "must_redact"})
    traces.append(trace)
    return QDTRuntimeEvaluationResult(
        metrics=tuple(sorted(metrics, key=lambda m: (m.metric_name, m.metric_id))),
        adapter_available=bool(adapter.available),
        integration_score=score,
        runtime_ready=bool(runtime_ready),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"evaluation_first": True, "live_qdt_write": False, "adapter_id": adapter.adapter_id, "threshold": opts.readiness_threshold},
    )
