"""QSPIN-PROD-7 source consideration matrix."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple
import json

class Prod7SourceFamily(str, Enum):
    QD6A = "QD6A"
    QSPIN8 = "QSPIN-8"
    PROD0 = "PROD-0"
    PROD1 = "PROD-1"
    PROD2 = "PROD-2"
    PROD3 = "PROD-3"
    PROD4 = "PROD-4"
    PROD5 = "PROD-5"
    PROD6 = "PROD-6"
    PROD7 = "PROD-7"
    TESTS = "tests"
    DOCS = "docs"
    RELEASE_MANIFESTS = "release_manifests"
    OPTIONAL_SKIPS = "optional_skips"

@dataclass(frozen=True)
class Prod7SourceRecord:
    family: Prod7SourceFamily
    direct_use: bool
    considered_not_touched: bool
    deferred_reason: str
    risk: str
    evidence: str
    structured_skip: bool = False
    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family.value, "direct_use": self.direct_use, "considered_not_touched": self.considered_not_touched, "deferred_reason": self.deferred_reason, "risk": self.risk, "evidence": self.evidence, "structured_skip": self.structured_skip}

@dataclass(frozen=True)
class Prod7SourceMatrix:
    records: Tuple[Prod7SourceRecord, ...]
    def to_dict(self) -> Dict[str, Any]: return {"records": [r.to_dict() for r in self.records]}
    def to_markdown(self) -> str:
        rows = ["| Family | Direct Use | Considered | Deferred Reason | Risk | Evidence |", "|---|---:|---:|---|---|---|"]
        for r in self.records:
            rows.append(f"| {r.family.value} | {r.direct_use} | {r.considered_not_touched} | {r.deferred_reason} | {r.risk} | {r.evidence} |")
        return "\n".join(rows) + "\n"

def build_prod7_source_matrix() -> Prod7SourceMatrix:
    records = []
    for fam in Prod7SourceFamily:
        direct = fam in {Prod7SourceFamily.PROD6, Prod7SourceFamily.PROD7, Prod7SourceFamily.TESTS, Prod7SourceFamily.DOCS, Prod7SourceFamily.RELEASE_MANIFESTS}
        records.append(Prod7SourceRecord(fam, direct, not direct, "preserved / no mutation" if not direct else "n/a", "low", "lineage recorded"))
    return Prod7SourceMatrix(tuple(records))

def export_prod7_source_matrix_markdown(matrix: Prod7SourceMatrix) -> str:
    return matrix.to_markdown()

def export_prod7_source_matrix_json(matrix: Prod7SourceMatrix) -> str:
    return json.dumps(matrix.to_dict(), indent=2, sort_keys=True)
