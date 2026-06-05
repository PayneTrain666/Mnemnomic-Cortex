"""Source consideration matrix for QSPIN-PROD-5."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


class ConsiderationStatus:
    USED = "USED"
    SKIPPED_MISSING = "SKIPPED_MISSING"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    EXCLUDED_SUPERSEDED = "EXCLUDED_SUPERSEDED"


@dataclass
class SourceRecord:
    source_id: str
    path: str
    status: str
    confidence: str
    risk: str
    use_decision: str
    exclusion_reason: str = ""
    exists: bool = False


DEFAULT_SOURCES = [
    ("QD6A", "/mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip", "active source of truth", "high", "low"),
    ("QSPIN-8", "/mnt/data/qspin8_qd6a_release_pack.zip", "lineage baseline", "medium", "low"),
    ("PROD-0", "/mnt/data/qspin_prod0_qd6a_release_pack.zip", "preserved contract lineage", "medium", "low"),
    ("PROD-1", "/mnt/data/qspin_prod1_qd6a_release_pack.zip", "preserved contract lineage", "medium", "low"),
    ("PROD-2", "/mnt/data/qspin_prod2_qd6a_release_pack.zip", "preserved contract lineage", "medium", "low"),
    ("PROD-3", "/mnt/data/qspin_prod3_qd6a_release_pack_refresh.zip", "refreshed production baseline", "high", "low"),
    ("PROD-4", "/mnt/data/qspin_prod4_qd6a_release_pack.zip", "immediate previous release", "high", "low"),
    ("PROD-4-PRINTOUT", "/mnt/data/qspin_prod4_full_printout.txt", "immediate previous full printout", "high", "low"),
    ("WM-7A", "/mnt/data/qdt_wm_maae_all_wm_zip_archives_bundle.zip", "historical comparison only", "medium", "medium"),
]


def build_source_matrix() -> List[SourceRecord]:
    records: List[SourceRecord] = []
    for source_id, path, decision, confidence, risk in DEFAULT_SOURCES:
        exists = Path(path).exists()
        if source_id == "WM-7A":
            status = ConsiderationStatus.REFERENCE_ONLY if exists else ConsiderationStatus.SKIPPED_MISSING
            exclusion = "Superseded by QD6A; reference only for rollback/delta audit."
        else:
            status = ConsiderationStatus.USED if exists else ConsiderationStatus.SKIPPED_MISSING
            exclusion = "" if exists else "Source file missing in /mnt/data; stage degrades safely with structured skip."
        records.append(SourceRecord(source_id, path, status, confidence, risk, decision, exclusion, exists))
    return records


def matrix_to_json(records: List[SourceRecord]) -> str:
    return json.dumps([asdict(r) for r in records], indent=2, sort_keys=True)


def matrix_to_markdown(records: List[SourceRecord]) -> str:
    lines = ["# QSPIN-PROD-5-QD6A Source Consideration Matrix", "", "| Source | Exists | Status | Confidence | Risk | Use decision | Exclusion / skip reason |", "|---|---:|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r.source_id} | {str(r.exists)} | {r.status} | {r.confidence} | {r.risk} | {r.use_decision} | {r.exclusion_reason} |")
    return "\n".join(lines) + "\n"


def consideration_coverage_audit(records: List[SourceRecord]) -> Dict[str, Any]:
    used = [r.source_id for r in records if r.status == ConsiderationStatus.USED]
    skipped = [r.source_id for r in records if r.status == ConsiderationStatus.SKIPPED_MISSING]
    reference = [r.source_id for r in records if r.status == ConsiderationStatus.REFERENCE_ONLY]
    return {
        "used": used,
        "skipped_missing": skipped,
        "reference_only": reference,
        "coverage_status": "PASS_WITH_STRUCTURED_SKIPS" if skipped else "PASS",
    }
