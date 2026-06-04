"""QSPIN-PROD-8 expanded release pack builder contracts."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Tuple
import json

class Prod8ReleaseMode(str, Enum):
    LOCAL_ARCHIVE_ONLY = "local_archive_only"

class Prod8ReleaseStatus(str, Enum):
    READY_TO_PACKAGE = "ready_to_package"
    BLOCKED = "blocked"

class Prod8ReleaseBlockReason(str, Enum):
    MISSING_CRITICAL_ARTIFACT = "missing_critical_artifact"
    EXTERNAL_UPLOAD_REQUESTED = "external_upload_requested"
    REPO_COMMIT_REQUESTED = "repo_commit_requested"

class Prod8ReleaseArtifactKind(str, Enum):
    MODULE = "module"
    TEST = "test"
    DOC = "doc"
    RUNNER = "runner"
    JSON_SUMMARY = "json_summary"
    MARKDOWN_SUMMARY = "markdown_summary"
    JUNIT_XML = "junit_xml"
    SOURCE_MATRIX = "source_matrix"
    READINESS_REVIEW = "readiness_review"
    BLOCKER_BURNDOWN = "blocker_burndown"
    PROBE_REPORT = "probe_report"
    CI_BASELINE = "ci_baseline"
    REMEDIATION_REGISTER = "remediation_register"
    OBSERVABILITY = "observability"
    AUDIT_SHIPCHECK = "audit_shipcheck"
    FINAL_HOLD_COMMAND = "final_hold_command"
    RELEASE_MANIFEST = "release_manifest"

@dataclass(frozen=True)
class Prod8ReleaseArtifactRecord:
    artifact_id: str
    kind: Prod8ReleaseArtifactKind
    path: str
    present: bool
    critical: bool = True

    def validate(self) -> "Prod8ReleaseArtifactRecord":
        if not self.artifact_id or not self.path:
            raise ValueError("artifact id and path required")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {"artifact_id": self.artifact_id, "kind": self.kind.value, "path": self.path, "present": self.present, "critical": self.critical}

@dataclass(frozen=True)
class Prod8ReleaseManifest:
    release_id: str
    stage: str
    artifacts: Tuple[Prod8ReleaseArtifactRecord, ...]
    safety_boundaries: Tuple[str, ...]
    final_hold_required: bool = True
    production_active: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {"release_id": self.release_id, "stage": self.stage, "artifacts": [a.to_dict() for a in self.artifacts], "safety_boundaries": list(self.safety_boundaries), "final_hold_required": self.final_hold_required, "production_active": self.production_active}

@dataclass(frozen=True)
class Prod8ReleaseBuildRequest:
    request_id: str
    artifacts: Tuple[Prod8ReleaseArtifactRecord, ...]
    external_upload_requested: bool = False
    repo_commit_requested: bool = False

@dataclass(frozen=True)
class Prod8ReleaseBuildResult:
    request_id: str
    status: Prod8ReleaseStatus
    block_reasons: Tuple[Prod8ReleaseBlockReason, ...]
    manifest: Prod8ReleaseManifest

    def to_dict(self) -> Dict[str, object]:
        return {"request_id": self.request_id, "status": self.status.value, "block_reasons": [r.value for r in self.block_reasons], "manifest": self.manifest.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

class Prod8ExpandedReleaseBuilder:
    def build(self, request: Prod8ReleaseBuildRequest) -> Prod8ReleaseBuildResult:
        if not request.request_id:
            raise ValueError("request_id required")
        reasons = []
        if request.external_upload_requested:
            reasons.append(Prod8ReleaseBlockReason.EXTERNAL_UPLOAD_REQUESTED)
        if request.repo_commit_requested:
            reasons.append(Prod8ReleaseBlockReason.REPO_COMMIT_REQUESTED)
        for artifact in request.artifacts:
            artifact.validate()
        if any(a.critical and not a.present for a in request.artifacts):
            reasons.append(Prod8ReleaseBlockReason.MISSING_CRITICAL_ARTIFACT)
        status = Prod8ReleaseStatus.BLOCKED if reasons else Prod8ReleaseStatus.READY_TO_PACKAGE
        manifest = Prod8ReleaseManifest(
            release_id="qspin_prod8_qd6a_release_pack",
            stage="QSPIN-PROD-8-QD6A",
            artifacts=tuple(sorted(request.artifacts, key=lambda a: a.path)),
            safety_boundaries=("no_live_routing", "no_payload_transfer", "no_writes", "no_commits", "no_production_activation"),
        )
        return Prod8ReleaseBuildResult(request.request_id, status, tuple(dict.fromkeys(reasons)), manifest)

def build_default_prod8_release_artifacts() -> Tuple[Prod8ReleaseArtifactRecord, ...]:
    items = [
        ("modules", Prod8ReleaseArtifactKind.MODULE, "mnemonic_cortex/working_memory/", True),
        ("tests", Prod8ReleaseArtifactKind.TEST, "tests/", True),
        ("docs", Prod8ReleaseArtifactKind.DOC, "docs/", True),
        ("runner", Prod8ReleaseArtifactKind.RUNNER, "prod8_manual_runner.py", True),
        ("source_matrix", Prod8ReleaseArtifactKind.SOURCE_MATRIX, "prod8_outputs/prod8_source_matrix.json", True),
        ("readiness", Prod8ReleaseArtifactKind.READINESS_REVIEW, "prod8_outputs/final_readiness_review.json", True),
        ("blockers", Prod8ReleaseArtifactKind.BLOCKER_BURNDOWN, "prod8_outputs/blocker_burndown.json", True),
        ("probe", Prod8ReleaseArtifactKind.PROBE_REPORT, "prod8_outputs/readonly_probe_report.json", True),
        ("ci", Prod8ReleaseArtifactKind.CI_BASELINE, "prod8_outputs/ci_baseline_freeze.json", True),
        ("remediation", Prod8ReleaseArtifactKind.REMEDIATION_REGISTER, "prod8_outputs/final_remediation_register.json", True),
        ("hold", Prod8ReleaseArtifactKind.FINAL_HOLD_COMMAND, "PROD8_FINAL_HOLD_COMMAND.md", True),
    ]
    return tuple(Prod8ReleaseArtifactRecord(*item) for item in items)
