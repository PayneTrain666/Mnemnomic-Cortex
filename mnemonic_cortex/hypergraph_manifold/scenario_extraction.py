"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: scenario extraction.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Scenario top-k extraction for HGM/HPME probability tensors.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from .config import HGMConfig
from .enums import TraceEventKind, ValidationSeverity
from .normalization import as_nested_payload
from .runtime_result import ScenarioCandidate, ScenarioExtractionResult
from .shapes import TensorShapeContract, iter_leaf_values, nested_shape
from .types import TraceRecord
from .validation import ValidationResult


def _metadata_tuple(metadata: Optional[Sequence[str]], size: int, prefix: str) -> Tuple[str, ...]:
    if metadata:
        values = tuple(str(x) for x in metadata)
        if len(values) >= size:
            return values
    return tuple(f"{prefix}_{i}" for i in range(size))


def extract_top_k_scenarios(
    payload: Any,
    k: int,
    contract: TensorShapeContract,
    config: HGMConfig = HGMConfig(),
    *,
    variable_ids: Optional[Sequence[str]] = None,
    magnitude_bin_ids: Optional[Sequence[str]] = None,
    depth_ids: Optional[Sequence[str]] = None,
    context_ids: Optional[Sequence[str]] = None,
    action_ids: Optional[Sequence[str]] = None,
) -> ScenarioExtractionResult:
    """Extract top-k probability cells as scenario candidates.

    Equal scores are ordered deterministically by their source index tuple.
    ``k <= 0`` returns an empty result with a warning trace instead of raising.
    """

    validation = ValidationResult()
    traces: List[TraceRecord] = []
    data = as_nested_payload(payload)

    if k <= 0:
        validation.warning("scenario_extraction.non_positive_k", "k <= 0; returning empty candidate set", "k")
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "scenario_extraction.extract_top_k_scenarios",
            severity=ValidationSeverity.WARNING,
            payload={"k": k, "reason": "non_positive_k"},
        ))
        return ScenarioExtractionResult(tuple(), top_k=0, validation=validation, trace_records=tuple(traces), metadata={"requested_k": k})

    try:
        shape = nested_shape(data)
        validation.merge(contract.validate_shape(shape))
    except ValueError as exc:
        validation.error("scenario_extraction.invalid_shape", str(exc), "payload")
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "scenario_extraction.extract_top_k_scenarios",
            severity=ValidationSeverity.ERROR,
            payload={"error": str(exc)},
        ))
        return ScenarioExtractionResult(tuple(), top_k=0, validation=validation, trace_records=tuple(traces), metadata={"requested_k": k})

    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "scenario_extraction.extract_top_k_scenarios",
            severity=ValidationSeverity.ERROR,
            payload={"shape": shape, "contract": contract.name},
        ))
        return ScenarioExtractionResult(tuple(), top_k=0, validation=validation, trace_records=tuple(traces), metadata={"requested_k": k})

    leaves = [(coords, float(value)) for coords, value in iter_leaf_values(data)]
    if not leaves:
        validation.warning("scenario_extraction.empty_payload", "empty payload; returning no candidates", "payload")
        traces.append(TraceRecord.create(
            TraceEventKind.VALIDATE,
            "scenario_extraction.extract_top_k_scenarios",
            severity=ValidationSeverity.WARNING,
            payload={"shape": shape, "reason": "empty_payload"},
        ))
        return ScenarioExtractionResult(tuple(), top_k=0, validation=validation, trace_records=tuple(traces), metadata={"requested_k": k})

    v_ids = _metadata_tuple(variable_ids, shape[0], "v")
    m_ids = _metadata_tuple(magnitude_bin_ids, shape[1], "m")
    d_ids = _metadata_tuple(depth_ids, shape[2] if len(shape) > 2 else 0, "d")
    c_ids = _metadata_tuple(context_ids, shape[3] if len(shape) > 3 else 0, "c")
    a_ids = _metadata_tuple(action_ids, shape[5] if len(shape) > 5 else 0, "a")

    ranked = sorted(leaves, key=lambda item: (-item[1], item[0]))
    top = ranked[: min(k, len(ranked))]
    candidates: List[ScenarioCandidate] = []
    for rank, (coords, value) in enumerate(top):
        trace = TraceRecord.create(
            TraceEventKind.CREATE,
            "scenario_extraction.extract_top_k_scenarios",
            payload={"rank": rank, "source_indices": coords, "probability": value},
        )
        traces.append(trace)
        candidates.append(ScenarioCandidate(
            candidate_id=f"scenario_{rank:04d}_{'_'.join(str(x) for x in coords)}",
            variable_id=v_ids[coords[0]],
            magnitude_bin_id=m_ids[coords[1]],
            depth=coords[2] if len(coords) > 2 else None,
            context_id=c_ids[coords[3]] if len(coords) > 3 and c_ids else None,
            time_index=coords[4] if len(coords) > 4 else None,
            action_id=a_ids[coords[5]] if len(coords) > 5 and a_ids else None,
            probability=value,
            score=value,
            source_indices=coords,
            trace_id=trace.trace_id,
        ))

    traces.append(TraceRecord.create(
        TraceEventKind.VALIDATE,
        "scenario_extraction.extract_top_k_scenarios",
        payload={"requested_k": k, "returned": len(candidates), "contract": contract.name},
    ))
    return ScenarioExtractionResult(
        candidates=tuple(candidates),
        top_k=len(candidates),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"requested_k": k, "available": len(leaves), "contract": contract.name},
    )
