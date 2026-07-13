"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: quality metrics.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Deterministic quality metrics for HGM-5 bridge evaluation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .enums import DepthLayer, GeometryType, TraceEventKind, ValidationSeverity
from .hgm4_result import BridgeExecutionPreview, HGMBridgePayload, SharedSlotLatticeHook
from .hgm5_result import BridgeEvaluationResult, BridgeQualityMetric, EmbeddingTrainerOptions
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


def _is_valid_depth(value: Any) -> bool:
    try:
        DepthLayer.coerce(value)
        return True
    except Exception:
        return False


def _is_valid_geometry(value: Any) -> bool:
    try:
        GeometryType.coerce(value)
        return True
    except Exception:
        return False


def _redaction_compatible(mapping: Any) -> bool:
    if not isinstance(mapping, Mapping):
        return True
    text = repr(mapping).lower()
    # Redacted metadata should not expose obvious high-risk keys with unredacted values.
    for term in _SECRET_TERMS:
        if term in text and "<redacted>" not in text:
            return False
    return True


def _make_metric(source_id: str, name: str, score: float, weight: float, explanation: str, trace_id: str) -> BridgeQualityMetric:
    return BridgeQualityMetric(
        metric_id=f"hgm5_metric_{_stable_hash(source_id, name)}",
        source_id=str(source_id),
        metric_name=name,
        score=_clamp01(score),
        weight=max(0.0, float(weight)),
        explanation=explanation,
        trace_id=trace_id,
    )


def _aggregate(metrics: Sequence[BridgeQualityMetric]) -> float:
    if not metrics:
        return 0.0
    total_w = sum(max(0.0, float(m.weight)) for m in metrics)
    if total_w <= 0:
        return 0.0
    return _clamp01(sum(float(m.score) * float(m.weight) for m in metrics) / total_w)


def evaluate_bridge_payload_quality(payloads: Sequence[HGMBridgePayload], config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> BridgeEvaluationResult:
    """Score bridge-payload quality using deterministic validation metrics."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    payload_tuple = tuple(payloads or tuple())
    if not payload_tuple:
        validation.warning("hgm5_payload_quality.empty", "no bridge payloads supplied", "payloads")
    if len(payload_tuple) > opts.max_records:
        validation.warning("hgm5_payload_quality.bounded", "payload count exceeded max_records; truncated for evaluation", "payloads")
    metrics: List[BridgeQualityMetric] = []
    for idx, payload in enumerate(payload_tuple[: opts.max_records]):
        if not isinstance(payload, HGMBridgePayload):
            validation.warning("hgm5_payload_quality.unsupported", "unsupported payload record skipped", f"payloads[{idx}]")
            continue
        trace = _trace("quality_metrics.evaluate_bridge_payload_quality", validation, {"source_id": payload.source_id, "metadata": payload.metadata})
        traces.append(trace)
        sid = payload.source_id or payload.payload_id or f"payload_{idx}"
        checks = [
            ("source_id_completeness", bool(payload.source_id), 1.0, "source_id is present"),
            ("depth_layer_validity", _is_valid_depth(payload.depth_layer), 1.0, "depth layer is valid"),
            ("geometry_validity", _is_valid_geometry(payload.geometry_type), 1.0, "geometry type is valid"),
            ("qspin_presence", bool(payload.qspin_signature_id), 0.8, "q-spin or placeholder is present"),
            ("content_summary_presence", bool(str(payload.content_summary or "").strip()), 0.8, "content summary is present"),
            ("trace_id_presence", bool(payload.trace_id), 0.6, "trace ID is present"),
            ("redaction_compatibility", _redaction_compatible(payload.metadata), 1.0, "metadata is redaction-compatible"),
        ]
        for name, ok, weight, explanation in checks:
            metrics.append(_make_metric(sid, name, 1.0 if ok else 0.0, weight, explanation, trace.trace_id))
    final_trace = _trace("quality_metrics.evaluate_bridge_payload_quality", validation, {"metric_count": len(metrics)})
    traces.append(final_trace)
    return BridgeEvaluationResult(
        metrics=tuple(metrics),
        aggregate_score=_aggregate(metrics),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"metric_count": len(metrics), "payload_count": min(len(payload_tuple), opts.max_records)},
    )


def evaluate_slot_hook_quality(hooks: Sequence[SharedSlotLatticeHook], config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> BridgeEvaluationResult:
    """Score shared slot-lattice hook quality."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    hook_tuple = tuple(hooks or tuple())
    if not hook_tuple:
        validation.warning("hgm5_hook_quality.empty", "no slot hooks supplied", "hooks")
    if len(hook_tuple) > opts.max_records:
        validation.warning("hgm5_hook_quality.bounded", "hook count exceeded max_records; truncated for evaluation", "hooks")
    metrics: List[BridgeQualityMetric] = []
    for idx, hook in enumerate(hook_tuple[: opts.max_records]):
        if not isinstance(hook, SharedSlotLatticeHook):
            validation.warning("hgm5_hook_quality.unsupported", "unsupported hook record skipped", f"hooks[{idx}]")
            continue
        trace = _trace("quality_metrics.evaluate_slot_hook_quality", validation, {"source_record_id": hook.source_record_id, "target_slot_id": hook.target_slot_id})
        traces.append(trace)
        sid = hook.source_record_id or hook.hook_id or f"hook_{idx}"
        checks = [
            ("source_id_completeness", bool(hook.source_record_id), 1.0, "source record ID is present"),
            ("target_slot_id_stability", bool(hook.target_slot_id and str(hook.target_slot_id).startswith("hgm_slot_")), 1.0, "target slot ID is stable"),
            ("depth_layer_validity", _is_valid_depth(hook.depth_layer), 1.0, "depth layer is valid"),
            ("geometry_validity", _is_valid_geometry(hook.geometry_type), 1.0, "geometry type is valid"),
            ("qspin_presence", bool(hook.qspin_signature_id), 0.8, "q-spin or placeholder is present"),
            ("dry_run_safety", bool(hook.dry_run) and not bool(hook.write_intent), 1.2, "hook is dry-run safe"),
            ("confidence_range", 0.0 <= float(hook.confidence) <= 1.0, 0.8, "confidence is in [0, 1]"),
        ]
        for name, ok, weight, explanation in checks:
            metrics.append(_make_metric(sid, name, 1.0 if ok else 0.0, weight, explanation, trace.trace_id))
    final_trace = _trace("quality_metrics.evaluate_slot_hook_quality", validation, {"metric_count": len(metrics)})
    traces.append(final_trace)
    return BridgeEvaluationResult(tuple(metrics), _aggregate(metrics), validation, tuple(traces), metadata={"metric_count": len(metrics), "hook_count": min(len(hook_tuple), opts.max_records)})


def evaluate_execution_preview_quality(preview: BridgeExecutionPreview, config=None, options: Optional[EmbeddingTrainerOptions | Mapping[str, Any]] = None) -> BridgeEvaluationResult:
    """Score preview-only bridge execution quality."""

    _ = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    metrics: List[BridgeQualityMetric] = []
    if not isinstance(preview, BridgeExecutionPreview):
        validation.warning("hgm5_preview_quality.invalid", "preview is missing or unsupported", "preview")
        trace = _trace("quality_metrics.evaluate_execution_preview_quality", validation, {"reason": "invalid_preview"})
        return BridgeEvaluationResult(tuple(), 0.0, validation, (trace,), metadata={"skipped": True})
    trace = _trace("quality_metrics.evaluate_execution_preview_quality", validation, {"preview_id": preview.preview_id, "metadata": preview.metadata})
    traces.append(trace)
    sid = preview.preview_id
    preview_only = bool(getattr(preview, "metadata", {}).get("preview_only", True)) and not bool(getattr(preview, "metadata", {}).get("executed", False))
    blocked_if_needed = True
    if preview.blocked_reason and preview.allowed:
        blocked_if_needed = False
    checks = [
        ("preview_only_behavior", preview_only, 1.5, "preview is marked preview-only and not executed"),
        ("blocked_write_intent", blocked_if_needed, 1.2, "write intent is blocked when needed"),
        ("planned_operation_count", len(preview.planned_operations) >= 0, 0.5, "planned operation count is available"),
        ("validation_status", bool(preview.validation.ok), 1.0, "preview validation is OK or warning-only"),
        ("no_execution_guarantee", getattr(preview, "metadata", {}).get("executed", False) is False, 1.5, "preview did not execute writes"),
    ]
    for name, ok, weight, explanation in checks:
        metrics.append(_make_metric(sid, name, 1.0 if ok else 0.0, weight, explanation, trace.trace_id))
    final_trace = _trace("quality_metrics.evaluate_execution_preview_quality", validation, {"metric_count": len(metrics)})
    traces.append(final_trace)
    return BridgeEvaluationResult(tuple(metrics), _aggregate(metrics), validation, tuple(traces), metadata={"metric_count": len(metrics), "operation_count": len(preview.planned_operations)})
