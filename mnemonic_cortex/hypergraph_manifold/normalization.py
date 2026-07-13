"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: normalization.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Normalization runtime for HGM/HPME probability payloads.

Supported modes:
- NONE: validate only, leave values unchanged.
- ROW_STOCHASTIC / ROW / MUTATION_AXIS: normalize along the mutation axis.
- GLOBAL_SUM / GLOBAL: normalize the full payload to sum 1.
- SOFTMAX: stable softmax over the mutation axis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

from .enums import ProbabilityNormalizationMode
from .runtime_result import NormalizationReport
from .shapes import TensorShapeContract, iter_leaf_values, nested_shape


ROW_MODES = {
    ProbabilityNormalizationMode.ROW,
    ProbabilityNormalizationMode.MUTATION_AXIS,
    ProbabilityNormalizationMode.ROW_STOCHASTIC,
}
GLOBAL_MODES = {ProbabilityNormalizationMode.GLOBAL, ProbabilityNormalizationMode.GLOBAL_SUM}


def as_nested_payload(payload: Any) -> Any:
    """Return a Python nested-list compatible payload when possible.

    Objects with ``tolist()`` such as numpy arrays and torch tensors are accepted
    without importing those libraries. Scalars and existing nested sequences are
    passed through recursively.
    """

    if hasattr(payload, "tolist") and not isinstance(payload, (list, tuple)):
        return payload.tolist()
    if isinstance(payload, tuple):
        return [as_nested_payload(item) for item in payload]
    if isinstance(payload, list):
        return [as_nested_payload(item) for item in payload]
    return payload


def total_sum(payload: Any) -> float:
    return float(sum(float(value) for _, value in iter_leaf_values(payload)))


def _zeros(shape: Sequence[int]) -> Any:
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(int(shape[0]))]


def _set_at(target: Any, coords: Tuple[int, ...], value: float) -> None:
    cursor = target
    for idx in coords[:-1]:
        cursor = cursor[idx]
    cursor[coords[-1]] = float(value)


def _build_from_flat(shape: Sequence[int], values: Dict[Tuple[int, ...], float]) -> Any:
    result = _zeros(tuple(shape))
    for coords, value in values.items():
        _set_at(result, coords, value)
    return result


def _group_key(coords: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
    return tuple(c for idx, c in enumerate(coords) if idx != axis)


def _row_normalize(payload: Any, contract: TensorShapeContract) -> Tuple[Any, Tuple[str, ...], Tuple[str, ...]]:
    shape = nested_shape(payload)
    axis = contract.mutation_axis_index
    leaves = [(coords, float(value)) for coords, value in iter_leaf_values(payload)]
    sums: Dict[Tuple[int, ...], float] = {}
    for coords, value in leaves:
        key = _group_key(coords, axis)
        sums[key] = sums.get(key, 0.0) + value

    warnings: List[str] = []
    errors: List[str] = []
    normalized: Dict[Tuple[int, ...], float] = {}
    for coords, value in leaves:
        key = _group_key(coords, axis)
        denom = sums.get(key, 0.0)
        if denom <= 0.0 or not math.isfinite(denom):
            errors.append(f"row group {key} cannot be normalized because sum={denom}")
            normalized[coords] = 0.0
        else:
            normalized[coords] = value / denom
    return _build_from_flat(shape, normalized), tuple(warnings), tuple(errors)


def _softmax(payload: Any, contract: TensorShapeContract) -> Tuple[Any, Tuple[str, ...], Tuple[str, ...]]:
    shape = nested_shape(payload)
    axis = contract.mutation_axis_index
    leaves = [(coords, float(value)) for coords, value in iter_leaf_values(payload)]
    groups: Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], float]]] = {}
    for coords, value in leaves:
        groups.setdefault(_group_key(coords, axis), []).append((coords, value))

    normalized: Dict[Tuple[int, ...], float] = {}
    errors: List[str] = []
    for key, items in groups.items():
        finite_items = [(coords, value) for coords, value in items if math.isfinite(value)]
        if len(finite_items) != len(items):
            errors.append(f"row group {key} contains non-finite values")
            for coords, _ in items:
                normalized[coords] = 0.0
            continue
        max_value = max(value for _, value in items)
        exps = [(coords, math.exp(value - max_value)) for coords, value in items]
        denom = sum(value for _, value in exps)
        if denom <= 0.0 or not math.isfinite(denom):
            errors.append(f"row group {key} cannot be softmax-normalized")
            for coords, _ in items:
                normalized[coords] = 0.0
        else:
            for coords, value in exps:
                normalized[coords] = value / denom
    return _build_from_flat(shape, normalized), tuple(), tuple(errors)


def normalize_probability_payload(
    payload: Any,
    mode: ProbabilityNormalizationMode,
    contract: Optional[TensorShapeContract] = None,
) -> Tuple[Any, NormalizationReport]:
    """Normalize a payload and return ``(normalized_payload, report)``.

    This function does not silently repair invalid rows. Zero-sum rows generate
    report errors and are filled with zeros so downstream code can remain
    structured and fail closed via the report/ValidationResult path.
    """

    mode = ProbabilityNormalizationMode.coerce(mode)
    data = as_nested_payload(payload)
    before = total_sum(data) if nested_shape(data) else float(data)

    if mode == ProbabilityNormalizationMode.NONE:
        after = total_sum(data) if nested_shape(data) else float(data)
        return data, NormalizationReport(mode=mode, before_sum=before, after_sum=after, repaired=False)

    if mode in GLOBAL_MODES:
        if before <= 0.0 or not math.isfinite(before):
            return data, NormalizationReport(
                mode=mode,
                before_sum=before,
                after_sum=before,
                repaired=False,
                errors=(f"global sum must be positive and finite, got {before}",),
            )
        shape = nested_shape(data)
        values = {coords: float(value) / before for coords, value in iter_leaf_values(data)}
        normalized = _build_from_flat(shape, values)
        return normalized, NormalizationReport(mode=mode, before_sum=before, after_sum=total_sum(normalized), repaired=False)

    if contract is None:
        return data, NormalizationReport(
            mode=mode,
            before_sum=before,
            after_sum=before,
            repaired=False,
            errors=("contract is required for mutation-axis normalization",),
        )

    if mode in ROW_MODES:
        normalized, warnings, errors = _row_normalize(data, contract)
        return normalized, NormalizationReport(
            mode=mode,
            before_sum=before,
            after_sum=total_sum(normalized),
            repaired=False,
            warnings=warnings,
            errors=errors,
        )

    if mode == ProbabilityNormalizationMode.SOFTMAX:
        normalized, warnings, errors = _softmax(data, contract)
        return normalized, NormalizationReport(
            mode=mode,
            before_sum=before,
            after_sum=total_sum(normalized),
            repaired=False,
            warnings=warnings,
            errors=errors,
        )

    return data, NormalizationReport(
        mode=mode,
        before_sum=before,
        after_sum=before,
        repaired=False,
        errors=(f"unsupported normalization mode {mode.value!r}",),
    )
