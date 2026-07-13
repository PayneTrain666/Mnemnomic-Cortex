"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-8 final source consideration matrix.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Tuple
import json

class Prod8SourceFamily(str, Enum):
    QD6A = "qd6a"
    QSPIN8 = "qspin8"
    PROD0 = "prod0"
    PROD1 = "prod1"
    PROD2 = "prod2"
    PROD3 = "prod3"
    PROD4 = "prod4"
    PROD5 = "prod5"
    PROD6 = "prod6"
    PROD7 = "prod7"
    PROD8 = "prod8"
    TESTS = "tests"
    DOCS = "docs"
    RELEASE_MANIFESTS = "release_manifests"
    OPTIONAL_SKIPS = "optional_skips"

@dataclass(frozen=True)
class Prod8SourceRecord:
    family: Prod8SourceFamily
    direct_use: bool
    considered_not_touched: bool
    deferred_reason: str
    risk: str
    evidence: str
    missing_optional: bool = False

    def validate(self) -> "Prod8SourceRecord":
        if not self.evidence:
            raise ValueError("source record requires evidence")
        if self.deferred_reason == "" and not self.direct_use and not self.considered_not_touched:
            raise ValueError("non-used source requires deferred reason or considered flag")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"family": self.family.value, "direct_use": self.direct_use, "considered_not_touched": self.considered_not_touched, "deferred_reason": self.deferred_reason, "risk": self.risk, "evidence": self.evidence, "missing_optional": self.missing_optional}

@dataclass(frozen=True)
class Prod8SourceMatrix:
    records: Tuple[Prod8SourceRecord, ...]

    def validate(self) -> "Prod8SourceMatrix":
        families = {r.family for r in self.records}
        missing = [f.value for f in Prod8SourceFamily if f not in families]
        if missing:
            raise ValueError("missing source families: " + ", ".join(missing))
        for rec in self.records:
            rec.validate()
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"records": [r.to_dict() for r in self.records]}


def build_prod8_source_matrix() -> Prod8SourceMatrix:
    records = []
    for fam in Prod8SourceFamily:
        records.append(Prod8SourceRecord(
            family=fam,
            direct_use=fam in {Prod8SourceFamily.PROD7, Prod8SourceFamily.PROD8, Prod8SourceFamily.TESTS, Prod8SourceFamily.DOCS, Prod8SourceFamily.RELEASE_MANIFESTS},
            considered_not_touched=fam not in {Prod8SourceFamily.PROD8},
            deferred_reason="Historical/lineage source considered but not modified." if fam not in {Prod8SourceFamily.PROD7, Prod8SourceFamily.PROD8, Prod8SourceFamily.TESTS, Prod8SourceFamily.DOCS, Prod8SourceFamily.RELEASE_MANIFESTS} else "",
            risk="Production activation risk remains blocked by final hold." if fam != Prod8SourceFamily.OPTIONAL_SKIPS else "Missing optional sources are structured skips.",
            evidence=f"PROD-8 matrix evidence for {fam.value}",
            missing_optional=fam == Prod8SourceFamily.OPTIONAL_SKIPS,
        ))
    return Prod8SourceMatrix(tuple(records)).validate()

def export_prod8_source_matrix_json(matrix: Prod8SourceMatrix) -> str:
    return json.dumps(matrix.validate().to_dict(), indent=2, sort_keys=True)

def export_prod8_source_matrix_markdown(matrix: Prod8SourceMatrix) -> str:
    lines = ["# PROD-8 Final Source Consideration Matrix", ""]
    for rec in matrix.validate().records:
        lines.append(f"- `{rec.family.value}`: direct_use={rec.direct_use}, considered_not_touched={rec.considered_not_touched}, risk={rec.risk}")
    return "\n".join(lines) + "\n"
