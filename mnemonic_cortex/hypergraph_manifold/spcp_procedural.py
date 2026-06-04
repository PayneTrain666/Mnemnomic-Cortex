"""Spherical–Projective Conformal Procedural Memory helpers for HGM-3."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .enums import TraceEventKind, ValidationSeverity
from .hgm3_result import (
    ProceduralActionSequence,
    SPCPEmbeddingResult,
    SPCPProcedureEmbedding,
    SPCPProceduralOptions,
    SPCPSimilarityResult,
)
from .shapes import finite_number
from .types import TraceRecord
from .validation import ValidationResult

_EPS = 1e-12


def _coerce_options(options: Optional[SPCPProceduralOptions | Mapping[str, Any]]) -> SPCPProceduralOptions:
    if options is None:
        return SPCPProceduralOptions()
    if isinstance(options, SPCPProceduralOptions):
        return options
    return SPCPProceduralOptions(**dict(options))


def _trace(component: str, validation: ValidationResult, payload=None) -> TraceRecord:
    return TraceRecord.create(
        TraceEventKind.VALIDATE,
        component,
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload=dict(payload or {}),
    )


def _clamp01(value: float) -> float:
    if not finite_number(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _normalize(values: Sequence[float], validation: ValidationResult, path: str, allow_zero: bool = False) -> Tuple[float, ...]:
    for idx, value in enumerate(values):
        if not finite_number(value):
            validation.error("hgm3_spcp.non_finite_state", "state vector values must be finite", f"{path}[{idx}]")
    if not validation.ok:
        return tuple()
    n = _norm(values)
    if n <= _EPS:
        if allow_zero:
            return tuple(0.0 for _ in values)
        validation.error("hgm3_spcp.zero_spherical_state", "spherical state requires non-zero norm", path)
        return tuple()
    return tuple(float(v) / n for v in values)


def _hash_scalar(text: str, modulus: int = 997) -> float:
    return float(sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text))) % modulus) / float(modulus)


def _sequence_numeric_features(sequence: ProceduralActionSequence, dimension: int) -> Tuple[float, ...]:
    values: List[float] = []
    values.append(float(len(sequence.primitives)))
    values.append(float(sequence.confidence))
    values.append(_hash_scalar(sequence.goal_label))
    for idx, primitive in enumerate(sequence.primitives):
        values.append(_hash_scalar(primitive.action_type))
        values.append(float(primitive.duration))
        values.append(float(primitive.confidence))
        params = dict(primitive.parameters or {})
        numeric = 0.0
        count = 0
        for val in params.values():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and finite_number(val):
                numeric += float(val)
                count += 1
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, (int, float)) and finite_number(item):
                        numeric += float(item)
                        count += 1
        values.append(numeric / max(1, count))
        values.append(_hash_scalar(primitive.object_id or primitive.frame_id or primitive.tool_id or idx))
    if len(values) < dimension:
        values.extend(0.0 for _ in range(dimension - len(values)))
    return tuple(values[:dimension])


def _projective_from_spherical(spherical: Sequence[float]) -> Tuple[float, ...]:
    # Alternating real/imag representation. The deterministic phase ramp gives
    # non-zero imaginary content while preserving a stable procedure signature.
    vals: List[float] = []
    for idx, value in enumerate(spherical):
        angle = (idx + 1) * math.pi / max(2, len(spherical))
        vals.append(float(value) * math.cos(angle))
        vals.append(float(value) * math.sin(angle))
    return tuple(vals)


def _bounded_warp(sequence: ProceduralActionSequence, dimension: int, bound: float) -> Tuple[float, ...]:
    seed = _hash_scalar(sequence.sequence_id, modulus=1009)
    values = []
    for idx in range(dimension):
        raw = math.sin((idx + 1) * (1.0 + seed))
        values.append(float(bound) * raw)
    return tuple(values)


def compute_spcp_procedure_embedding(
    sequence: ProceduralActionSequence,
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> SPCPEmbeddingResult:
    """Build a dependency-light spherical/projective/conformal embedding."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    if not isinstance(sequence, ProceduralActionSequence):
        validation.error("hgm3_spcp.invalid_sequence", "sequence must be ProceduralActionSequence", "sequence")
        trace = _trace("spcp_procedural.compute_spcp_procedure_embedding", validation, {"reason": "invalid_sequence"})
        return SPCPEmbeddingResult(None, validation, (trace,), metadata={"similarity_ready": False})
    raw = _sequence_numeric_features(sequence, opts.embedding_dimension)
    spherical = _normalize(raw, validation, "spherical_state", opts.allow_zero_spherical_fallback)
    if not validation.ok:
        trace = _trace("spcp_procedural.compute_spcp_procedure_embedding", validation, {"sequence_id": sequence.sequence_id})
        return SPCPEmbeddingResult(None, validation, (trace,), metadata={"similarity_ready": False})
    projective = _projective_from_spherical(spherical)
    warp = _bounded_warp(sequence, len(spherical), opts.conformal_warp_bound)
    for idx, value in enumerate(warp):
        if not finite_number(value) or abs(float(value)) > opts.conformal_warp_bound + 1e-12:
            validation.error("hgm3_spcp.warp_out_of_bounds", "conformal warp must be finite and bounded", f"conformal_warp[{idx}]")
    trace = _trace("spcp_procedural.compute_spcp_procedure_embedding", validation, {"sequence_id": sequence.sequence_id})
    emb = SPCPProcedureEmbedding(
        embedding_id=f"hgm3_spcp_{sequence.sequence_id}",
        sequence_id=sequence.sequence_id,
        spherical_state=spherical,
        projective_state=projective,
        conformal_warp=warp,
        similarity_ready=validation.ok,
        trace_id=trace.trace_id,
        metadata={"spcp": "spherical_projective_conformal", "warp_bound": opts.conformal_warp_bound},
    )
    return SPCPEmbeddingResult(emb if validation.ok else None, validation, (trace,), metadata={"similarity_ready": validation.ok})


def _as_complex(values: Sequence[float]) -> Tuple[complex, ...]:
    if len(values) >= 2 and len(values) % 2 == 0:
        return tuple(complex(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2))
    return tuple(complex(float(v), 0.0) for v in values)


def _cosine_similarity(a: Sequence[float], b: Sequence[float], validation: ValidationResult) -> float:
    if len(a) != len(b):
        validation.error("hgm3_spcp.dimension_mismatch", "spherical vectors must have same dimension", "spherical_state")
        return 0.0
    na, nb = _norm(a), _norm(b)
    if na <= _EPS or nb <= _EPS:
        validation.error("hgm3_spcp.zero_norm", "spherical similarity requires non-zero vectors", "spherical_state")
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return _clamp01((dot / max(_EPS, na * nb) + 1.0) / 2.0)


def _projective_similarity(a: Sequence[float], b: Sequence[float], validation: ValidationResult) -> float:
    ca, cb = _as_complex(a), _as_complex(b)
    if len(ca) != len(cb):
        validation.error("hgm3_spcp.projective_dimension_mismatch", "projective vectors differ after complex pairing", "projective_state")
        return 0.0
    na = math.sqrt(sum(abs(x) ** 2 for x in ca))
    nb = math.sqrt(sum(abs(x) ** 2 for x in cb))
    if na <= _EPS or nb <= _EPS:
        validation.error("hgm3_spcp.projective_zero_norm", "projective similarity requires non-zero vectors", "projective_state")
        return 0.0
    inner = sum(x.conjugate() * y for x, y in zip(ca, cb))
    return _clamp01(abs(inner) / max(_EPS, na * nb))


def spcp_procedure_similarity(
    embedding_a: SPCPProcedureEmbedding,
    embedding_b: SPCPProcedureEmbedding,
    config=None,
) -> SPCPSimilarityResult:
    """Compute phase-insensitive SPCP procedure similarity."""

    validation = ValidationResult()
    if not isinstance(embedding_a, SPCPProcedureEmbedding) or not isinstance(embedding_b, SPCPProcedureEmbedding):
        validation.error("hgm3_spcp.invalid_embedding", "both inputs must be SPCPProcedureEmbedding", "embedding")
        trace = _trace("spcp_procedural.spcp_procedure_similarity", validation, {"reason": "invalid_embedding"})
        return SPCPSimilarityResult(0.0, 1.0, validation, (trace,), metadata={})
    if not embedding_a.similarity_ready or not embedding_b.similarity_ready:
        validation.error("hgm3_spcp.not_similarity_ready", "embeddings must be similarity_ready", "embedding")
    spherical = _cosine_similarity(embedding_a.spherical_state, embedding_b.spherical_state, validation)
    projective = _projective_similarity(embedding_a.projective_state, embedding_b.projective_state, validation)
    warp_penalty = 0.0
    if len(embedding_a.conformal_warp) == len(embedding_b.conformal_warp) and embedding_a.conformal_warp:
        warp_penalty = min(0.1, _norm([a - b for a, b in zip(embedding_a.conformal_warp, embedding_b.conformal_warp)]))
    similarity = _clamp01(0.45 * spherical + 0.50 * projective + 0.05 * (1.0 - warp_penalty)) if validation.ok else 0.0
    distance = max(0.0, 1.0 - similarity)
    trace = _trace("spcp_procedural.spcp_procedure_similarity", validation, {
        "sequence_a": embedding_a.sequence_id,
        "sequence_b": embedding_b.sequence_id,
    })
    return SPCPSimilarityResult(similarity, distance, validation, (trace,), metadata={"spherical": spherical, "projective": projective, "warp_penalty": warp_penalty})
