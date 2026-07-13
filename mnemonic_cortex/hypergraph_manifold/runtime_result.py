"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: runtime result.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Runtime result dataclasses for HGM/HPME probability expansion.

HGM-0B keeps runtime outputs explicit and audit-friendly. The classes in this
module intentionally avoid torch/numpy dependencies so they can be used in
validation, tests, manifests, and future adapter layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .enums import ProbabilityNormalizationMode
from .shapes import TensorShapeContract
from .types import TraceRecord
from .validation import ValidationResult


@dataclass(frozen=True)
class NormalizationReport:
    """Report emitted by normalization routines."""

    mode: ProbabilityNormalizationMode
    before_sum: float
    after_sum: float
    repaired: bool = False
    warnings: Tuple[str, ...] = tuple()
    errors: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ProbabilityNormalizationMode.coerce(self.mode))
        object.__setattr__(self, "warnings", tuple(self.warnings or tuple()))
        object.__setattr__(self, "errors", tuple(self.errors or tuple()))


@dataclass(frozen=True)
class ProbabilityExpansionResult:
    """High-level output of HGM-0B probability expansion."""

    payload: Any
    contract: Optional[TensorShapeContract]
    normalization_mode: ProbabilityNormalizationMode
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    normalization_report: Optional[NormalizationReport] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalization_mode", ProbabilityNormalizationMode.coerce(self.normalization_mode))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))


@dataclass(frozen=True)
class ScenarioCandidate:
    """One probability cell extracted as a scenario candidate."""

    candidate_id: str
    variable_id: str
    magnitude_bin_id: str
    probability: float
    score: float
    source_indices: Tuple[int, ...]
    trace_id: str
    depth: Optional[int] = None
    context_id: Optional[str] = None
    time_index: Optional[int] = None
    action_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_indices", tuple(int(x) for x in self.source_indices))


@dataclass(frozen=True)
class ScenarioExtractionResult:
    """Top-k scenario extraction output."""

    candidates: Tuple[ScenarioCandidate, ...]
    top_k: int
    validation: ValidationResult
    trace_records: Tuple[TraceRecord, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates or tuple()))
        object.__setattr__(self, "trace_records", tuple(self.trace_records or tuple()))
