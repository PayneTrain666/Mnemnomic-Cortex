"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-8 read-only runtime probe report generator.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Mapping, Tuple
import json

class ProbeReportMode(str, Enum):
    READ_ONLY_REPORT = "read_only_report"

class ProbeReportStatus(str, Enum):
    GENERATED = "generated"
    GENERATED_WITH_FINDINGS = "generated_with_findings"

class ProbeReportFindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)
class ProbeReportFinding:
    finding_id: str
    severity: ProbeReportFindingSeverity
    title: str
    detail: str
    carry_forward: bool = False

    def validate(self) -> "ProbeReportFinding":
        if not self.finding_id or not self.title:
            raise ValueError("finding requires id and title")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"finding_id": self.finding_id, "severity": self.severity.value, "title": self.title, "detail": self.detail, "carry_forward": self.carry_forward}

@dataclass(frozen=True)
class ProbeReportSection:
    section_id: str
    title: str
    status: str
    findings: Tuple[ProbeReportFinding, ...] = ()

    def validate(self) -> "ProbeReportSection":
        if not self.section_id or not self.title:
            raise ValueError("section requires id and title")
        for finding in self.findings:
            finding.validate()
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"section_id": self.section_id, "title": self.title, "status": self.status, "findings": [f.to_dict() for f in self.findings]}

@dataclass(frozen=True)
class ProbeReport:
    report_id: str
    mode: ProbeReportMode
    status: ProbeReportStatus
    sections: Tuple[ProbeReportSection, ...]
    residual_risk: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {"report_id": self.report_id, "mode": self.mode.value, "status": self.status.value, "sections": [s.to_dict() for s in self.sections], "residual_risk": list(self.residual_risk)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# PROD-8 Read-Only Runtime Probe Report", "", f"Status: `{self.status.value}`", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append(f"Status: `{section.status}`")
            for finding in section.findings:
                lines.append(f"- **{finding.severity.value}** `{finding.finding_id}`: {finding.title}")
            lines.append("")
        lines.append("## Residual Risk")
        lines.extend(f"- {risk}" for risk in self.residual_risk)
        return "\n".join(lines) + "\n"

class ReadOnlyRuntimeProbeReportGenerator:
    def generate(self, sections: Tuple[ProbeReportSection, ...] | None = None) -> ProbeReport:
        sections = sections or build_default_probe_report_sections()
        for section in sections:
            section.validate()
        findings = [f for section in sections for f in section.findings]
        status = ProbeReportStatus.GENERATED_WITH_FINDINGS if findings else ProbeReportStatus.GENERATED
        risk = ("Live runtime was not called; real operational behavior remains unvalidated.", "All probe evidence remains read-only/pre-activation.")
        return ProbeReport("prod8_readonly_probe_report", ProbeReportMode.READ_ONLY_REPORT, status, sections, risk)

def build_default_probe_report_sections() -> Tuple[ProbeReportSection, ...]:
    def sec(section_id: str, title: str, status: str, finding: str = "") -> ProbeReportSection:
        findings = (ProbeReportFinding(section_id + "-F1", ProbeReportFindingSeverity.INFO, finding, finding, True),) if finding else ()
        return ProbeReportSection(section_id, title, status, findings)
    return (
        sec("source_import", "Source/import probe results", "read_only_pass"),
        sec("dataclass_contract", "Dataclass contract probe results", "read_only_pass"),
        sec("enum_contract", "Enum contract probe results", "read_only_pass"),
        sec("manifest", "Manifest probe results", "read_only_pass"),
        sec("release_metadata", "Release metadata probe results", "read_only_pass"),
        sec("no_write", "No-write sentinel results", "blocked"),
        sec("no_network", "No-network sentinel results", "blocked"),
        sec("no_commit", "No-commit sentinel results", "blocked"),
        sec("no_live_route", "No-live-route sentinel results", "blocked"),
        sec("no_payload_transfer", "No-payload-transfer sentinel results", "blocked"),
        sec("structured_skip", "Structured skip results", "available", "Optional missing sources were recorded as skips, not hidden passes."),
    )
