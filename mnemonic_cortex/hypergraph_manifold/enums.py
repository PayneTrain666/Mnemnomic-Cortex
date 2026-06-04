"""Enumerations for HGM/HPME foundation types.

HGM-0A is intentionally dependency-light. These enums use ``str`` values
to keep serialized traces, JSON manifests, and future config files stable.
"""

from __future__ import annotations

from enum import Enum


class _CoercibleEnum(str, Enum):
    """String enum with explicit, fail-closed coercion.

    ``Enum(value)`` already validates exact values. ``coerce`` also accepts
    case-insensitive names and values while refusing unknown inputs.
    """

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for item in cls:
                if normalized in {item.value.lower(), item.name.lower()}:
                    return item
        raise ValueError(f"Invalid {cls.__name__}: {value!r}")


class GeometryType(_CoercibleEnum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"
    SPHERICAL = "spherical"
    TORUS = "torus"
    COMPLEX_PROJECTIVE = "complex_projective"
    GRASSMANN = "grassmann"
    FISHER_SIMPLEX = "fisher_simplex"
    PRODUCT = "product"
    SPCP = "spcp"


class MutationDirection(_CoercibleEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    CATEGORICAL_SHIFT = "categorical_shift"
    UNKNOWN = "unknown"


class HyperedgeKind(_CoercibleEnum):
    MUTATION = "mutation"
    SCENARIO = "scenario"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    ANALOGY = "analogy"
    CONFLICT = "conflict"
    OPPORTUNITY = "opportunity"


class ProbabilityNormalizationMode(_CoercibleEnum):
    NONE = "none"
    GLOBAL = "global"  # HGM-0A compatibility alias
    GLOBAL_SUM = "global_sum"
    ROW = "row"  # HGM-0A compatibility alias
    MUTATION_AXIS = "mutation_axis"  # HGM-0A compatibility alias
    ROW_STOCHASTIC = "row_stochastic"
    SOFTMAX = "softmax"


class ValidationSeverity(_CoercibleEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceEventKind(_CoercibleEnum):
    CREATE = "create"
    VALIDATE = "validate"
    NORMALIZE = "normalize"
    REPAIR = "repair"
    FAIL = "fail"
    SERIALIZE = "serialize"


class DepthLayer(Enum):
    """Canonical eight-depth lattice used by HGM/HPME.

    D0 observation, D1 mutation, D2 entity, D3 relation, D4 causal,
    D5 procedural, D6 counterfactual, D7 strategic.
    """

    D0_OBSERVATION = 0
    D1_MUTATION = 1
    D2_ENTITY = 2
    D3_RELATION = 3
    D4_CAUSAL = 4
    D5_PROCEDURAL = 5
    D6_COUNTERFACTUAL = 6
    D7_STRATEGIC = 7

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            for item in cls:
                if item.value == value:
                    return item
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                return cls.coerce(int(raw))
            normalized = raw.lower()
            for item in cls:
                if normalized in {item.name.lower(), f"d{item.value}", str(item.value)}:
                    return item
        raise ValueError(f"Invalid DepthLayer: {value!r}")

    @classmethod
    def is_valid_value(cls, value) -> bool:
        try:
            cls.coerce(value)
            return True
        except ValueError:
            return False
