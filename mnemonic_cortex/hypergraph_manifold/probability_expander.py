"""Hyperset Probability Expander runtime for HGM/HPME.

This module is the HGM-0B runtime spine. It converts typed mutation tokens or
matrix-like inputs into validated probability payloads, applies explicit
normalization, and keeps trace records for auditability.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

from .config import HGMConfig
from .enums import ProbabilityNormalizationMode, TraceEventKind, ValidationSeverity
from .normalization import as_nested_payload, normalize_probability_payload
from .runtime_result import ProbabilityExpansionResult
from .shapes import (
    P_VM,
    P_VMD,
    P_VMDC,
    P_VMDCT,
    P_VMDCTA,
    TensorShapeContract,
    finite_number,
    iter_leaf_values,
    nested_shape,
)
from .types import MagnitudeBin, MutationToken, TraceRecord
from .validation import ValidationResult


RANK_CONTRACTS: Dict[int, TensorShapeContract] = {
    2: P_VM,
    3: P_VMD,
    4: P_VMDC,
    5: P_VMDCT,
    6: P_VMDCTA,
}


def infer_probability_shape_contract(payload: Any) -> Tuple[Optional[TensorShapeContract], ValidationResult]:
    """Infer HGM/HPME probability tensor contract from rank.

    Supported ranks map as follows:
    2 -> P[v,m]
    3 -> P[v,m,d]
    4 -> P[v,m,d,c]
    5 -> P[v,m,d,c,t]
    6 -> P[v,m,d,c,t,a]
    """

    result = ValidationResult()
    data = as_nested_payload(payload)
    try:
        shape = nested_shape(data)
    except ValueError as exc:
        result.error("probability_shape.ragged", str(exc), "payload")
        return None, result
    rank = len(shape)
    contract = RANK_CONTRACTS.get(rank)
    if contract is None:
        result.error(
            "probability_shape.unsupported_rank",
            f"unsupported probability rank {rank}; expected one of {sorted(RANK_CONTRACTS)}",
            "payload",
        )
        return None, result
    result.merge(contract.validate_shape(shape))
    return contract, result


def validate_probability_payload(
    payload: Any,
    contract: TensorShapeContract,
    config: HGMConfig = HGMConfig(),
    *,
    normalization_mode: ProbabilityNormalizationMode = ProbabilityNormalizationMode.NONE,
) -> ValidationResult:
    """Validate numeric payload against a shape contract and probability rules."""

    result = ValidationResult()
    mode = ProbabilityNormalizationMode.coerce(normalization_mode)
    data = as_nested_payload(payload)
    try:
        shape = nested_shape(data)
        result.merge(contract.validate_shape(shape))
    except ValueError as exc:
        result.error("probability_payload.ragged", str(exc), "payload")
        return result

    if not result.ok:
        return result
    try:
        _ = contract.mutation_axis_index
    except ValueError as exc:
        result.error("probability_payload.missing_mutation_axis", str(exc), "contract")
        return result

    # Fully inspect nested Python payloads and objects converted via tolist().
    for coords, value in iter_leaf_values(data):
        path = "payload" + "".join(f"[{i}]" for i in coords)
        if not finite_number(value):
            result.error("probability_payload.non_finite", "probability value must be finite", path)
            continue
        numeric = float(value)
        if numeric < 0.0 and mode != ProbabilityNormalizationMode.SOFTMAX:
            result.error("probability_payload.negative", "probability value must be non-negative before normalization", path)
    return result


def _default_ids(size: int, prefix: str) -> Tuple[str, ...]:
    return tuple(f"{prefix}_{i}" for i in range(size))


def expand_from_mutation_tokens(
    tokens: Iterable[MutationToken],
    config: HGMConfig = HGMConfig(),
) -> ProbabilityExpansionResult:
    """Convert typed mutation tokens into a P[v,m] probability matrix."""

    validation = ValidationResult()
    traces: List[TraceRecord] = []
    token_list = list(tokens)
    if not token_list:
        validation.error("mutation_expansion.empty_tokens", "at least one MutationToken is required", "tokens")
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "probability_expander.expand_from_mutation_tokens",
            severity=ValidationSeverity.ERROR,
            payload={"reason": "empty_tokens"},
        ))
        return ProbabilityExpansionResult(
            payload=[],
            contract=P_VM,
            normalization_mode=ProbabilityNormalizationMode.NONE,
            validation=validation,
            trace_records=tuple(traces),
            metadata={"variable_ids": tuple(), "magnitude_bin_ids": tuple()},
        )

    variable_ids: List[str] = []
    magnitude_ids: List[str] = []
    for index, token in enumerate(token_list):
        token_result = token.validate(config)
        validation.merge(token_result)
        if token.probability is None:
            validation.error("mutation_expansion.missing_probability", "token probability is required for expansion", f"tokens[{index}].probability")
        if token.variable_id and token.variable_id not in variable_ids:
            variable_ids.append(token.variable_id)
        if token.magnitude_bin_id and token.magnitude_bin_id not in magnitude_ids:
            magnitude_ids.append(token.magnitude_bin_id)

    v_index = {value: idx for idx, value in enumerate(variable_ids)}
    m_index = {value: idx for idx, value in enumerate(magnitude_ids)}
    matrix = [[0.0 for _ in magnitude_ids] for _ in variable_ids]
    for token in token_list:
        if token.variable_id in v_index and token.magnitude_bin_id in m_index and token.probability is not None and finite_number(token.probability):
            matrix[v_index[token.variable_id]][m_index[token.magnitude_bin_id]] += float(token.probability)

    traces.append(TraceRecord.create(
        TraceEventKind.CREATE,
        "probability_expander.expand_from_mutation_tokens",
        payload={"token_count": len(token_list), "variables": len(variable_ids), "magnitudes": len(magnitude_ids)},
    ))
    return ProbabilityExpansionResult(
        payload=matrix,
        contract=P_VM,
        normalization_mode=ProbabilityNormalizationMode.NONE,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "variable_ids": tuple(variable_ids),
            "magnitude_bin_ids": tuple(magnitude_ids),
            "magnitude_bins": tuple(MagnitudeBin(mid, float(i), float(i), mid) for i, mid in enumerate(magnitude_ids)),
        },
    )


def build_probability_expansion(
    payload_or_tokens: Any,
    config: HGMConfig = HGMConfig(),
    normalization_mode: ProbabilityNormalizationMode = ProbabilityNormalizationMode.NONE,
) -> ProbabilityExpansionResult:
    """High-level HGM-0B expansion entry point."""

    mode = ProbabilityNormalizationMode.coerce(normalization_mode)
    traces: List[TraceRecord] = []

    if isinstance(payload_or_tokens, (list, tuple)) and all(isinstance(item, MutationToken) for item in payload_or_tokens):
        base = expand_from_mutation_tokens(payload_or_tokens, config)
        validation = ValidationResult.combine([base.validation])
        payload = base.payload
        contract = base.contract
        traces.extend(base.trace_records)
    else:
        payload = as_nested_payload(payload_or_tokens)
        contract, inferred = infer_probability_shape_contract(payload)
        validation = ValidationResult.combine([inferred])
        if contract is None:
            traces.append(TraceRecord.create(
                TraceEventKind.FAIL,
                "probability_expander.build_probability_expansion",
                severity=ValidationSeverity.ERROR,
                payload={"reason": "shape_inference_failed"},
            ))
            return ProbabilityExpansionResult(
                payload=payload,
                contract=None,
                normalization_mode=mode,
                validation=validation,
                trace_records=tuple(traces),
                metadata={},
            )

    if contract is None:
        validation.error("probability_expansion.missing_contract", "could not infer or construct a contract", "contract")
        return ProbabilityExpansionResult(payload=payload, contract=None, normalization_mode=mode, validation=validation, trace_records=tuple(traces), metadata={})

    validation.merge(validate_probability_payload(payload, contract, config, normalization_mode=mode))
    if not validation.ok:
        traces.append(TraceRecord.create(
            TraceEventKind.FAIL,
            "probability_expander.build_probability_expansion",
            severity=ValidationSeverity.ERROR,
            payload={"contract": contract.name, "mode": mode.value},
        ))
        return ProbabilityExpansionResult(payload=payload, contract=contract, normalization_mode=mode, validation=validation, trace_records=tuple(traces), metadata={})

    normalized, report = normalize_probability_payload(payload, mode, contract)
    for err in report.errors:
        validation.error("probability_normalization.error", err, "normalization")
    for warning in report.warnings:
        validation.warning("probability_normalization.warning", warning, "normalization")

    if validation.ok:
        validation.merge(validate_probability_payload(normalized, contract, config, normalization_mode=ProbabilityNormalizationMode.NONE))

    traces.append(TraceRecord.create(
        TraceEventKind.NORMALIZE,
        "probability_expander.build_probability_expansion",
        severity=ValidationSeverity.INFO if validation.ok else ValidationSeverity.ERROR,
        payload={"contract": contract.name, "mode": mode.value, "before_sum": report.before_sum, "after_sum": report.after_sum},
    ))
    metadata = {
        "shape": nested_shape(normalized) if validation.ok else nested_shape(as_nested_payload(payload)),
        "contract": contract.name,
        "lineage_tags": config.lineage_tags,
    }
    return ProbabilityExpansionResult(
        payload=normalized,
        contract=contract,
        normalization_mode=mode,
        validation=validation,
        trace_records=tuple(traces),
        metadata=metadata,
        normalization_report=report,
    )
