"""Integration readiness scoring for HGM-5."""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Mapping, Optional, Sequence

from .embedding_trainer import build_baseline_hgm_embeddings
from .enums import TraceEventKind, ValidationSeverity
from .hgm4_result import BridgeExecutionPreview, HGMBridgePayload, SharedSlotLatticeHook, TraceSafeMemoryPlan
from .hgm5_result import (
    BridgeEvaluationResult,
    EmbeddingTrainerOptions,
    HGM5EmbeddingEvaluationResult,
    IntegrationScore,
    IntegrationScoringResult,
)
from .quality_metrics import evaluate_bridge_payload_quality, evaluate_execution_preview_quality, evaluate_slot_hook_quality
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def _stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _coerce_options(options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]]) -> EmbeddingTrainerOptions:
    if options is None:
        return EmbeddingTrainerOptions()
    if isinstance(options, EmbeddingTrainerOptions):
        return options
    return EmbeddingTrainerOptions(**dict(options))


def _redact_value(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_redact_value(key, v) for v in value)
    return value


def _trace(component: str, validation: ValidationResult, payload: Optional[Mapping[str, Any]] = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={str(k): _redact_value(str(k), v) for k, v in dict(payload or {}).items()},
    )


def _clamp01(value: float) -> float:
    try:
        if not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _aggregate(scores: Sequence[IntegrationScore]) -> float:
    if not scores:
        return 0.0
    return _clamp01(sum(float(s.score) * float(s.confidence) for s in scores) / max(1e-9, sum(float(s.confidence) for s in scores)))


def _records_from_plan_or_records(records_or_plan: Any) -> tuple[Any, ...]:
    if records_or_plan is None:
        return tuple()
    if isinstance(records_or_plan, TraceSafeMemoryPlan):
        return tuple(records_or_plan.bridge_payloads) + tuple(records_or_plan.slot_hooks)
    if isinstance(records_or_plan, (list, tuple)):
        return tuple(records_or_plan)
    return (records_or_plan,)


def _extract_bridge_components(records_or_plan: Any):
    if isinstance(records_or_plan, TraceSafeMemoryPlan):
        return tuple(records_or_plan.bridge_payloads), tuple(records_or_plan.slot_hooks), None
    if isinstance(records_or_plan, BridgeExecutionPreview):
        return tuple(), tuple(), records_or_plan
    records = _records_from_plan_or_records(records_or_plan)
    payloads = tuple(r for r in records if isinstance(r, HGMBridgePayload))
    hooks = tuple(r for r in records if isinstance(r, SharedSlotLatticeHook))
    previews = tuple(r for r in records if isinstance(r, BridgeExecutionPreview))
    return payloads, hooks, previews[0] if previews else None


def score_hgm_integration_readiness(records_or_plan: Any, config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> IntegrationScoringResult:
    """Combine embedding, payload, hook, and preview metrics into readiness scores."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    records = _records_from_plan_or_records(records_or_plan)
    if not records:
        validation.warning("hgm5_integration.empty", "no records supplied for integration scoring", "records_or_plan")
    if len(records) > opts.max_records:
        validation.warning("hgm5_integration.bounded", "record count exceeded max_records; truncated", "records_or_plan")
    records = records[: opts.max_records]

    embeddings = build_baseline_hgm_embeddings(records, config=config, options=opts)
    validation.merge(embeddings.validation)
    traces.extend(embeddings.trace_records)
    payloads, hooks, preview = _extract_bridge_components(records_or_plan)
    payload_eval = evaluate_bridge_payload_quality(payloads, config=config, options=opts)
    hook_eval = evaluate_slot_hook_quality(hooks, config=config, options=opts)
    validation.merge(payload_eval.validation).merge(hook_eval.validation)
    traces.extend(payload_eval.trace_records)
    traces.extend(hook_eval.trace_records)
    scores: List[IntegrationScore] = []
    trace = _trace("integration_scoring.score_hgm_integration_readiness", validation, {"record_count": len(records)})
    traces.append(trace)
    scores.append(IntegrationScore(
        score_id=f"hgm5_score_{_stable_hash('embeddings', len(embeddings.embeddings))}",
        source_id="embedding_trainer",
        source_type="EmbeddingTrainerResult",
        score=_clamp01(1.0 if embeddings.embeddings else 0.0),
        confidence=0.8,
        explanation="baseline embeddings generated" if embeddings.embeddings else "no baseline embeddings generated",
        trace_id=trace.trace_id,
    ))
    scores.append(IntegrationScore(
        score_id=f"hgm5_score_{_stable_hash('payload_eval', payload_eval.aggregate_score)}",
        source_id="bridge_payloads",
        source_type="BridgeEvaluationResult",
        score=payload_eval.aggregate_score,
        confidence=0.9 if payload_eval.metrics else 0.3,
        explanation="bridge payload quality aggregate",
        trace_id=trace.trace_id,
    ))
    scores.append(IntegrationScore(
        score_id=f"hgm5_score_{_stable_hash('hook_eval', hook_eval.aggregate_score)}",
        source_id="slot_hooks",
        source_type="BridgeEvaluationResult",
        score=hook_eval.aggregate_score,
        confidence=0.9 if hook_eval.metrics else 0.3,
        explanation="slot hook quality aggregate",
        trace_id=trace.trace_id,
    ))
    if preview is not None:
        preview_eval = evaluate_execution_preview_quality(preview, config=config, options=opts)
        validation.merge(preview_eval.validation)
        traces.extend(preview_eval.trace_records)
        scores.append(IntegrationScore(
            score_id=f"hgm5_score_{_stable_hash('preview_eval', preview_eval.aggregate_score)}",
            source_id=preview.preview_id,
            source_type="BridgeExecutionPreview",
            score=preview_eval.aggregate_score,
            confidence=0.9,
            explanation="execution preview quality aggregate",
            trace_id=trace.trace_id,
        ))
    scores.sort(key=lambda s: (s.source_type, s.source_id, s.score_id))
    final_trace = _trace("integration_scoring.score_hgm_integration_readiness", validation, {"score_count": len(scores)})
    traces.append(final_trace)
    return IntegrationScoringResult(tuple(scores), _aggregate(scores), validation, tuple(traces), metadata={"score_count": len(scores), "record_count": len(records)})


def _empty_bridge_eval(validation: ValidationResult, traces: Sequence[TraceRecord]) -> BridgeEvaluationResult:
    return BridgeEvaluationResult(tuple(), 0.0, validation, tuple(traces), metadata={"empty": True})


def build_hgm5_embedding_evaluation(records_or_plan: Any, config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> HGM5EmbeddingEvaluationResult:
    """High-level HGM-5 entry point."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    records = _records_from_plan_or_records(records_or_plan)
    if not records:
        validation.warning("hgm5_build.empty", "no records supplied for HGM-5 evaluation", "records_or_plan")
    trainer = build_baseline_hgm_embeddings(records, config=config, options=opts)
    validation.merge(trainer.validation)
    traces.extend(trainer.trace_records)
    payloads, hooks, preview = _extract_bridge_components(records_or_plan)
    payload_eval = evaluate_bridge_payload_quality(payloads, config=config, options=opts)
    hook_eval = evaluate_slot_hook_quality(hooks, config=config, options=opts)
    validation.merge(payload_eval.validation).merge(hook_eval.validation)
    traces.extend(payload_eval.trace_records)
    traces.extend(hook_eval.trace_records)
    # HGM-5 returns one bridge_evaluation_result; keep payload as primary but include hook/preview in metadata.
    preview_eval = None
    if preview is not None:
        preview_eval = evaluate_execution_preview_quality(preview, config=config, options=opts)
        validation.merge(preview_eval.validation)
        traces.extend(preview_eval.trace_records)
    integration = score_hgm_integration_readiness(records_or_plan, config=config, options=opts)
    validation.merge(integration.validation)
    traces.extend(integration.trace_records)
    final_trace = _trace("integration_scoring.build_hgm5_embedding_evaluation", validation, {"record_count": len(records), "aggregate_score": integration.aggregate_score})
    traces.append(final_trace)
    bridge_eval = BridgeEvaluationResult(
        metrics=tuple(payload_eval.metrics) + tuple(hook_eval.metrics) + (tuple(preview_eval.metrics) if preview_eval else tuple()),
        aggregate_score=_clamp01((payload_eval.aggregate_score + hook_eval.aggregate_score + (preview_eval.aggregate_score if preview_eval else 0.0)) / (3 if preview_eval else 2 if (payload_eval.metrics or hook_eval.metrics) else 1)),
        validation=validation,
        trace_records=tuple(payload_eval.trace_records) + tuple(hook_eval.trace_records) + (tuple(preview_eval.trace_records) if preview_eval else tuple()),
        metadata={
            "payload_aggregate": payload_eval.aggregate_score,
            "hook_aggregate": hook_eval.aggregate_score,
            "preview_aggregate": preview_eval.aggregate_score if preview_eval else None,
        },
    )
    return HGM5EmbeddingEvaluationResult(
        trainer_result=trainer,
        bridge_evaluation_result=bridge_eval,
        integration_scoring_result=integration,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"evaluation_first": True, "integration_readiness": integration.aggregate_score, "record_count": len(records)},
    )
