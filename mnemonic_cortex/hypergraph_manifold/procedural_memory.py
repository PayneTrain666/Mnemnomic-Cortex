"""Dependency-light procedural memory store/retrieval for HGM-3."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .action_sequence import validate_action_sequence
from .enums import TraceEventKind, ValidationSeverity
from .hgm3_result import (
    ProceduralActionSequence,
    ProceduralMemoryRetrievalCandidate,
    ProceduralMemoryRetrievalResult,
    ProceduralMemoryStoreResult,
    SPCPProcedureEmbedding,
    SPCPProceduralOptions,
)
from .spcp_procedural import compute_spcp_procedure_embedding, spcp_procedure_similarity
from .types import TraceRecord
from .validation import ValidationResult


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


def store_procedural_sequences(
    sequences: Sequence[ProceduralActionSequence],
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> ProceduralMemoryStoreResult:
    """Validate and embed procedural sequences without mutating global state."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    stored_sequences: List[ProceduralActionSequence] = []
    stored_embeddings: List[SPCPProcedureEmbedding] = []
    seen = set()

    if not sequences:
        validation.warning("hgm3_store.empty_sequences", "empty sequence input; no procedural records stored", "sequences")
        trace = _trace("procedural_memory.store_procedural_sequences", validation, {"reason": "empty_sequences"})
        traces.append(trace)
        return ProceduralMemoryStoreResult(tuple(), tuple(), validation, tuple(traces), metadata={"stored_count": 0})

    for idx, sequence in enumerate(sequences):
        seq_validation = validate_action_sequence(sequence, opts)
        validation.merge(seq_validation)
        if not seq_validation.ok:
            continue
        if sequence.sequence_id in seen:
            validation.error("hgm3_store.duplicate_sequence", f"duplicate sequence_id {sequence.sequence_id!r}", f"sequences[{idx}].sequence_id")
            continue
        seen.add(sequence.sequence_id)
        emb_result = compute_spcp_procedure_embedding(sequence, config=config, options=opts)
        validation.merge(emb_result.validation)
        traces.extend(emb_result.trace_records)
        if emb_result.embedding is not None and emb_result.validation.ok:
            stored_sequences.append(sequence)
            stored_embeddings.append(emb_result.embedding)

    trace = _trace("procedural_memory.store_procedural_sequences", validation, {"stored_count": len(stored_sequences)})
    traces.append(trace)
    return ProceduralMemoryStoreResult(tuple(stored_sequences), tuple(stored_embeddings), validation, tuple(traces), metadata={"stored_count": len(stored_sequences)})


def _query_embedding(query, config=None, options=None):
    if isinstance(query, SPCPProcedureEmbedding):
        return query, ValidationResult(), tuple()
    if isinstance(query, ProceduralActionSequence):
        result = compute_spcp_procedure_embedding(query, config=config, options=options)
        return result.embedding, result.validation, result.trace_records
    validation = ValidationResult()
    validation.error("hgm3_retrieve.invalid_query", "query must be ProceduralActionSequence or SPCPProcedureEmbedding", "query")
    trace = _trace("procedural_memory.retrieve_similar_procedures", validation, {"reason": "invalid_query"})
    return None, validation, (trace,)


def retrieve_similar_procedures(
    query_sequence_or_embedding,
    stored_embeddings: Sequence[SPCPProcedureEmbedding],
    top_k: int,
    config=None,
    options: Optional[SPCPProceduralOptions | Mapping[str, Any]] = None,
) -> ProceduralMemoryRetrievalResult:
    """Retrieve nearest procedural records by SPCP similarity."""

    opts = _coerce_options(options)
    validation = ValidationResult()
    traces: List[TraceRecord] = []
    query_embedding, query_validation, query_traces = _query_embedding(query_sequence_or_embedding, config=config, options=opts)
    validation.merge(query_validation)
    traces.extend(query_traces)

    if int(top_k) <= 0:
        validation.warning("hgm3_retrieve.non_positive_top_k", "top_k <= 0; returning empty retrieval result", "top_k")
        trace = _trace("procedural_memory.retrieve_similar_procedures", validation, {"top_k": top_k})
        traces.append(trace)
        return ProceduralMemoryRetrievalResult(tuple(), validation, tuple(traces), metadata={"top_k": top_k, "available": len(stored_embeddings or tuple())})

    if query_embedding is None or not validation.ok:
        trace = _trace("procedural_memory.retrieve_similar_procedures", validation, {"reason": "invalid_query"})
        traces.append(trace)
        return ProceduralMemoryRetrievalResult(tuple(), validation, tuple(traces), metadata={"top_k": top_k})

    embeddings = tuple(stored_embeddings or tuple())
    if not embeddings:
        validation.warning("hgm3_retrieve.empty_store", "stored embedding list is empty", "stored_embeddings")
        trace = _trace("procedural_memory.retrieve_similar_procedures", validation, {"reason": "empty_store"})
        traces.append(trace)
        return ProceduralMemoryRetrievalResult(tuple(), validation, tuple(traces), metadata={"top_k": top_k, "available": 0})

    scored: List[ProceduralMemoryRetrievalCandidate] = []
    for idx, emb in enumerate(embeddings):
        if not isinstance(emb, SPCPProcedureEmbedding):
            validation.error("hgm3_retrieve.invalid_embedding", "stored_embeddings must contain SPCPProcedureEmbedding records", f"stored_embeddings[{idx}]")
            continue
        sim = spcp_procedure_similarity(query_embedding, emb, config=config)
        validation.merge(sim.validation)
        traces.extend(sim.trace_records)
        if sim.validation.ok:
            scored.append(ProceduralMemoryRetrievalCandidate(
                candidate_id=f"hgm3_retrieval_{idx:04d}_{emb.sequence_id}",
                sequence_id=emb.sequence_id,
                similarity=sim.similarity,
                distance=sim.distance,
                confidence=sim.similarity,
                source_trace_id=emb.trace_id,
                metadata={"rank_source_index": idx},
            ))
    scored.sort(key=lambda item: (-item.similarity, item.distance, item.sequence_id, item.candidate_id))
    bounded_k = min(int(top_k), len(scored))
    trace = _trace("procedural_memory.retrieve_similar_procedures", validation, {"top_k": top_k, "returned": bounded_k})
    traces.append(trace)
    return ProceduralMemoryRetrievalResult(tuple(scored[:bounded_k]), validation, tuple(traces), metadata={"top_k": bounded_k, "available": len(scored)})
