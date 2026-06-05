"""QSPIN-PROD-6 source consideration matrix."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

class Prod6SourceFamily(str, Enum):
    QD6A = "QD6A"
    QSPIN8 = "QSPIN-8"
    PROD0 = "PROD-0"
    PROD1 = "PROD-1"
    PROD2 = "PROD-2"
    PROD3 = "PROD-3"
    PROD4 = "PROD-4"
    PROD5 = "PROD-5"
    PROD6 = "PROD-6"
    TESTS = "tests"
    DOCS = "docs"
    RELEASE_MANIFESTS = "release_manifests"
    OPTIONAL_SKIPS = "optional_skips"

@dataclass(frozen=True)
class Prod6SourceRecord:
    family: Prod6SourceFamily
    directly_used: bool
    considered_not_touched: bool
    deferred_reason: str
    risk: str
    evidence: str
    structured_skip: bool = False

@dataclass(frozen=True)
class Prod6SourceMatrix:
    records: Tuple[Prod6SourceRecord, ...]

    def validate(self) -> "Prod6SourceMatrix":
        have = {r.family for r in self.records}
        missing = set(Prod6SourceFamily) - have
        if missing:
            raise ValueError("source matrix missing families: " + ",".join(sorted(m.value for m in missing)))
        return self

    def to_json_dict(self) -> Dict[str, object]:
        return {"records": [{"family": r.family.value, "directly_used": r.directly_used, "considered_not_touched": r.considered_not_touched, "deferred_reason": r.deferred_reason, "risk": r.risk, "evidence": r.evidence, "structured_skip": r.structured_skip} for r in self.records]}


def build_prod6_source_matrix() -> Prod6SourceMatrix:
    records = []
    for family in Prod6SourceFamily:
        records.append(Prod6SourceRecord(
            family=family,
            directly_used=family in {Prod6SourceFamily.PROD5, Prod6SourceFamily.PROD6, Prod6SourceFamily.TESTS, Prod6SourceFamily.DOCS, Prod6SourceFamily.RELEASE_MANIFESTS},
            considered_not_touched=family in {Prod6SourceFamily.QD6A, Prod6SourceFamily.QSPIN8, Prod6SourceFamily.PROD0, Prod6SourceFamily.PROD1, Prod6SourceFamily.PROD2, Prod6SourceFamily.PROD3, Prod6SourceFamily.PROD4},
            deferred_reason="live runtime remains blocked; source considered for compatibility" if family not in {Prod6SourceFamily.PROD5, Prod6SourceFamily.PROD6} else "active synthetic stress/replay source",
            risk="low-synthetic-only",
            evidence="PROD-6 matrix, tests, runner output, release manifest",
            structured_skip=family is Prod6SourceFamily.OPTIONAL_SKIPS,
        ))
    return Prod6SourceMatrix(tuple(records)).validate()


def export_prod6_source_matrix_markdown(matrix: Prod6SourceMatrix | None = None) -> str:
    matrix = matrix or build_prod6_source_matrix()
    lines = ["# QSPIN-PROD-6 Source Consideration Matrix", "", "| Family | Direct Use | Considered Not Touched | Risk | Evidence |", "|---|---:|---:|---|---|"]
    for r in matrix.records:
        lines.append(f"| {r.family.value} | {r.directly_used} | {r.considered_not_touched} | {r.risk} | {r.evidence} |")
    return "\n".join(lines) + "\n"


def export_prod6_source_matrix_json(matrix: Prod6SourceMatrix | None = None) -> Dict[str, object]:
    return (matrix or build_prod6_source_matrix()).to_json_dict()
