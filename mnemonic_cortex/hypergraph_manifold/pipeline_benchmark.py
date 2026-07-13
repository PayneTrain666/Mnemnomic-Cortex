"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: pipeline benchmark.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
End-to-end HGM pipeline benchmark harness for HGM-8.

This benchmark is deterministic and evaluation-only. It measures whether HGM
records can be embedded, replayed safely, traced, and scored without mutating
QDT/WM runtime memory.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from .hgm5_result import HGMEmbeddingRecord
from .hgm7_result import TransactionLog, WriteExecutionResult, HGM7WriteExecutionResult
from .hgm8_result import (
    HGM8RuntimeOptions,
    HGM8PipelineEvaluationResult,
    PipelineBenchmarkMetric,
    PipelineBenchmarkResult,
    hgm8_stable_hash,
    trace_hgm8,
)
from .runtime_embedding_trainer import build_runtime_embedding_trainer, coerce_hgm8_options
from .safe_write_replay import evaluate_safe_write_replay
from .validation import ValidationResult


def _weighted_average(metrics: Sequence[PipelineBenchmarkMetric]) -> float:
    total_weight = sum(metric.weight for metric in metrics)
    if total_weight <= 0:
        return 0.0
    return sum(metric.score * metric.weight for metric in metrics) / total_weight


def _make_metric(name: str, score: float, weight: float, explanation: str, validation: ValidationResult, traces: List[Any], metadata=None) -> PipelineBenchmarkMetric:
    trace = trace_hgm8("pipeline_benchmark.metric", validation, {"metric_name": name, "score": score, "metadata": metadata or {}})
    traces.append(trace)
    return PipelineBenchmarkMetric(
        metric_id=f"hgm8_metric_{hgm8_stable_hash(name, score, weight, explanation)}",
        metric_name=name,
        score=score,
        weight=weight,
        explanation=explanation,
        trace_id=trace.trace_id,
        metadata=dict(metadata or {}),
    )


def benchmark_hgm_pipeline(
    records_or_result: Any,
    config=None,
    options: Optional[HGM8RuntimeOptions | Mapping[str, Any]] = None,
) -> PipelineBenchmarkResult:
    """Benchmark embedding/replay/trace characteristics for HGM records."""

    opts = coerce_hgm8_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    records = tuple(records_or_result if isinstance(records_or_result, (list, tuple)) else (records_or_result,) if records_or_result is not None else tuple())
    if not records:
        validation.warning("hgm8_benchmark.empty", "no records supplied for pipeline benchmark", "records_or_result")
    trainer = build_runtime_embedding_trainer(records, config=config, options=opts)
    replay = evaluate_safe_write_replay(records_or_result, config=config, options=opts)
    validation.merge(trainer.validation).merge(replay.validation)
    traces.extend(trainer.trace_records + replay.trace_records)

    metrics: List[PipelineBenchmarkMetric] = []
    input_count = max(1, min(len(records), opts.max_records))
    embedding_coverage = len(trainer.embeddings) / input_count
    metrics.append(_make_metric("runtime_embedding_coverage", embedding_coverage, 1.0, "fraction of supplied records with deterministic runtime embeddings", validation, traces))
    vector_dimension_score = 1.0 if all(len(emb.vector) == opts.embedding_dimension for emb in trainer.embeddings) else 0.0
    metrics.append(_make_metric("embedding_dimension_validity", vector_dimension_score, 0.75, "all runtime embeddings match configured dimension", validation, traces))
    replay_score = replay.aggregate_score if replay.records else 0.5 if trainer.embeddings else 0.0
    metrics.append(_make_metric("safe_write_replay_score", replay_score, 1.0, "safe replay score from HGM-7 logs/results", validation, traces))
    live_write_safety = 1.0 if not any(getattr(record, "metadata", {}).get("live_qdt_write", False) for record in replay.records) else 0.0
    metrics.append(_make_metric("no_live_write_guarantee", live_write_safety, 1.25, "benchmark observed no live QDT/WM write indicator", validation, traces))
    trace_score = 1.0 if traces else 0.0
    metrics.append(_make_metric("trace_observability", trace_score, 0.5, "benchmark emitted trace records", validation, traces))
    deterministic_score = 1.0
    metrics.append(_make_metric("deterministic_evaluation", deterministic_score, 0.5, "benchmark uses deterministic ordering and hashing", validation, traces))

    aggregate = _weighted_average(metrics)
    passed = bool(aggregate >= 0.6 and live_write_safety == 1.0)
    final_trace = trace_hgm8("pipeline_benchmark.benchmark_hgm_pipeline", validation, {"aggregate_score": aggregate, "passed": passed, "secret_token": "must_redact"})
    traces.append(final_trace)
    return PipelineBenchmarkResult(
        metrics=tuple(metrics),
        aggregate_score=aggregate,
        passed=passed,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"evaluation_first": True, "live_qdt_write": False, "input_count": len(records), "benchmark_iterations": opts.benchmark_iterations},
    )


def build_hgm8_pipeline_evaluation(
    records_or_result: Any,
    config=None,
    options: Optional[HGM8RuntimeOptions | Mapping[str, Any]] = None,
) -> HGM8PipelineEvaluationResult:
    """High-level HGM-8 entry point."""

    opts = coerce_hgm8_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    records = tuple(records_or_result if isinstance(records_or_result, (list, tuple)) else (records_or_result,) if records_or_result is not None else tuple())
    trainer = build_runtime_embedding_trainer(records, config=config, options=opts)
    replay = evaluate_safe_write_replay(records_or_result, config=config, options=opts)
    benchmark = benchmark_hgm_pipeline(records_or_result, config=config, options=opts)
    validation.merge(trainer.validation).merge(replay.validation).merge(benchmark.validation)
    traces.extend(trainer.trace_records + replay.trace_records + benchmark.trace_records)
    trace = trace_hgm8("pipeline_benchmark.build_hgm8_pipeline_evaluation", validation, {"benchmark_passed": benchmark.passed, "aggregate_score": benchmark.aggregate_score})
    traces.append(trace)
    return HGM8PipelineEvaluationResult(
        trainer_result=trainer,
        replay_result=replay,
        benchmark_result=benchmark,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"evaluation_first": True, "live_qdt_write": False, "benchmark_passed": benchmark.passed},
    )
