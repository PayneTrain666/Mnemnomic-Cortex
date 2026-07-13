"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: bridge payloads.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Trace-safe bridge payload construction for HGM-4.

This module converts HGM records into small, redacted, lineage-preserving
payloads suitable for dry-run QDT/WM bridge planning.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Tuple

from .enums import DepthLayer, GeometryType, TraceEventKind, ValidationSeverity
from .hgm1_result import BoundScenarioHyperedge
from .hgm2_result import DepthRetrievalTarget, ManifoldRouteAssignment
from .hgm3_result import (
    ProceduralActionSequence,
    ProceduralMemoryRetrievalCandidate,
    RoboticsPlanningActionOption,
    SPCPProcedureEmbedding,
)
from .hgm4_result import BridgePayloadBuildResult, HGMBridgePayload, QDTWMBridgeOptions
from .types import TraceRecord
from .validation import ValidationResult

_SECRET_TERMS = ("secret", "token", "api_key", "password", "credential", "private_key")


def _stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _coerce_options(options: Optional[QDTWMBridgeOptions | Mapping[str, Any]]) -> QDTWMBridgeOptions:
    if options is None:
        return QDTWMBridgeOptions()
    if isinstance(options, QDTWMBridgeOptions):
        return options
    return QDTWMBridgeOptions(**dict(options))


def _trace(component: str, validation: ValidationResult, payload: Optional[Mapping[str, Any]] = None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload=_redact_mapping(dict(payload or {})),
    )


def _redact_value(key: str, value: Any) -> Any:
    if any(term in str(key).lower() for term in _SECRET_TERMS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return _redact_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_redact_value(key, item) for item in value)
    return value


def _redact_mapping(mapping: Mapping[str, Any]) -> dict:
    return {str(k): _redact_value(str(k), v) for k, v in dict(mapping or {}).items()}


def _summary(text: str, max_len: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _metadata(record: Any) -> Mapping[str, Any]:
    meta = getattr(record, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        return _redact_mapping(meta)
    return {"metadata_repr": _summary(repr(meta), 256)}


def _maybe_qspin(record: Any, opts: QDTWMBridgeOptions, source_id: str) -> str:
    meta = getattr(record, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        for key in ("qspin_signature_id", "qspin", "q_spin", "qspin_id"):
            if meta.get(key):
                return str(meta[key])
    direct = getattr(record, "qspin_signature_id", None)
    if direct:
        return str(direct)
    return f"{opts.qspin_placeholder_prefix}_{_stable_hash(source_id)}"


def _record_content_summary(record: Any, source_type: str, opts: QDTWMBridgeOptions) -> str:
    if isinstance(record, BoundScenarioHyperedge):
        return _summary(
            f"BoundScenarioHyperedge {record.hyperedge_id}: candidates={len(record.candidate_ids)} nodes={len(record.node_ids)} coherence={record.coherence_score:.4f}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, ManifoldRouteAssignment):
        return _summary(
            f"ManifoldRouteAssignment {record.assignment_id}: hyperedge={record.hyperedge_id} chart={record.chart_id} geometry={record.geometry_type.value} depth={record.depth_layer.name}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, DepthRetrievalTarget):
        return _summary(
            f"DepthRetrievalTarget {record.target_id}: hyperedge={record.hyperedge_id} depth={record.depth_layer.name} key={record.retrieval_key}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, ProceduralActionSequence):
        return _summary(
            f"ProceduralActionSequence {record.sequence_id}: primitives={len(record.primitives)} goal={record.goal_label}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, SPCPProcedureEmbedding):
        return _summary(
            f"SPCPProcedureEmbedding {record.embedding_id}: sequence={record.sequence_id} spherical={len(record.spherical_state)} projective={len(record.projective_state)}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, ProceduralMemoryRetrievalCandidate):
        return _summary(
            f"ProceduralMemoryRetrievalCandidate {record.candidate_id}: sequence={record.sequence_id} similarity={record.similarity:.4f}",
            opts.max_payload_summary_length,
        )
    if isinstance(record, RoboticsPlanningActionOption):
        return _summary(
            f"RoboticsPlanningActionOption {record.option_id}: sequence={record.sequence_id} goal={record.expected_goal} advisory_only=True",
            opts.max_payload_summary_length,
        )
    return _summary(f"Unsupported HGM bridge record type {source_type}", opts.max_payload_summary_length)


def _payload_fields(record: Any, opts: QDTWMBridgeOptions) -> Tuple[str, str, DepthLayer, GeometryType]:
    if isinstance(record, BoundScenarioHyperedge):
        depth = DepthLayer.D3_RELATION
        geom = GeometryType.coerce(record.metadata.get("geometry_type", record.metadata.get("geometry", opts.default_geometry_type))) if isinstance(record.metadata, Mapping) else opts.default_geometry_type
        return "BoundScenarioHyperedge", record.hyperedge_id, depth, geom
    if isinstance(record, ManifoldRouteAssignment):
        return "ManifoldRouteAssignment", record.assignment_id, record.depth_layer, record.geometry_type
    if isinstance(record, DepthRetrievalTarget):
        geom = GeometryType.coerce(record.metadata.get("geometry_type", opts.default_geometry_type)) if isinstance(record.metadata, Mapping) else opts.default_geometry_type
        return "DepthRetrievalTarget", record.target_id, record.depth_layer, geom
    if isinstance(record, ProceduralActionSequence):
        return "ProceduralActionSequence", record.sequence_id, DepthLayer.D5_PROCEDURAL, GeometryType.SPCP
    if isinstance(record, SPCPProcedureEmbedding):
        return "SPCPProcedureEmbedding", record.embedding_id, DepthLayer.D5_PROCEDURAL, GeometryType.SPCP
    if isinstance(record, ProceduralMemoryRetrievalCandidate):
        return "ProceduralMemoryRetrievalCandidate", record.candidate_id, DepthLayer.D5_PROCEDURAL, GeometryType.SPCP
    if isinstance(record, RoboticsPlanningActionOption):
        return "RoboticsPlanningActionOption", record.option_id, DepthLayer.D5_PROCEDURAL, GeometryType.SPCP
    raise TypeError(f"Unsupported HGM bridge record type: {type(record).__name__}")


def build_bridge_payload_from_hgm_record(record: Any, config=None, options: Optional[QDTWMBridgeOptions | Mapping[str, Any]] = None) -> BridgePayloadBuildResult:
    """Convert a supported HGM record into a trace-safe bridge payload.

    Unsupported records return a structured validation warning rather than
    raising. The returned payload is ``None`` in that case.
    """

    opts = _coerce_options(options)
    validation = ValidationResult()
    try:
        source_type, source_id, depth_layer, geometry_type = _payload_fields(record, opts)
    except Exception as exc:
        validation.warning("hgm4_payload.unsupported_record", f"unsupported bridge record: {type(record).__name__}", "record")
        trace = _trace("bridge_payloads.build_bridge_payload_from_hgm_record", validation, {"record_type": type(record).__name__, "reason": str(exc)})
        return BridgePayloadBuildResult(None, validation, (trace,), metadata={"skipped": True, "record_type": type(record).__name__})

    if not source_id:
        validation.error("hgm4_payload.missing_source_id", "source record ID must be stable and non-empty", "source_id")
    qspin = _maybe_qspin(record, opts, source_id)
    trace = _trace(
        "bridge_payloads.build_bridge_payload_from_hgm_record",
        validation,
        {"source_type": source_type, "source_id": source_id, "qspin_signature_id": qspin},
    )
    payload = None
    if validation.ok:
        payload_id = f"hgm4_payload_{_stable_hash(source_type, source_id, depth_layer.name, geometry_type.value)}"
        metadata = _metadata(record)
        if qspin.startswith(opts.qspin_placeholder_prefix):
            metadata = dict(metadata)
            metadata["qspin_placeholder_generated"] = True
        payload = HGMBridgePayload(
            payload_id=payload_id,
            source_type=source_type,
            source_id=str(source_id),
            depth_layer=depth_layer,
            geometry_type=geometry_type,
            content_summary=_record_content_summary(record, source_type, opts),
            qspin_signature_id=qspin,
            trace_id=trace.trace_id,
            metadata=metadata,
        )
    return BridgePayloadBuildResult(payload, validation, (trace,), metadata={"source_type": source_type, "source_id": str(source_id)})
