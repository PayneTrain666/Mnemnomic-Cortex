"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: release consolidation.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Release manifest consolidation for HGM v0.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .api_freeze import coerce_hgm10_options
from .hgm10_result import HGM10ReleaseOptions, ReleaseConsolidationRecord, ReleaseManifestSummary, hgm10_stable_hash, trace_hgm10
from .validation import ValidationResult


def _manifest_field(data: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _summarize_manifest(path: Path, root: Path, validation: ValidationResult) -> ReleaseManifestSummary:
    stage_id = path.parent.name
    data: Mapping[str, Any] = {}
    found = path.exists()
    release_name = stage_id
    version = ""
    file_count = 0
    test_summary = "not recorded"
    if found:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            release_name = str(_manifest_field(data, "release_name", "name", "stage", "stage_id", default=stage_id))
            version = str(_manifest_field(data, "version", "release_version", default=""))
            files_value = _manifest_field(data, "files", "changed_files", "artifacts", default=[])
            if isinstance(files_value, Mapping):
                file_count = len(files_value)
            elif isinstance(files_value, (list, tuple)):
                file_count = len(files_value)
            else:
                file_count = 0
            tests_value = _manifest_field(data, "tests", "test_summary", "pytest", default="")
            test_summary = str(tests_value) if tests_value else "manifest present"
        except Exception as exc:  # pragma: no cover - defensive manifest parsing
            validation.warning("hgm10_release.manifest_parse_failed", f"Could not parse {path}: {exc}", str(path))
    else:
        validation.warning("hgm10_release.manifest_missing", f"Manifest missing: {path}", str(path))
    trace = trace_hgm10("release_consolidation.manifest_summary", validation, {"stage_id": stage_id, "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path)})
    return ReleaseManifestSummary(
        summary_id=f"hgm10_manifest_{hgm10_stable_hash(stage_id, str(path))}",
        stage_id=stage_id,
        manifest_path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        found=found,
        release_name=release_name,
        version=version,
        file_count=file_count,
        test_summary=test_summary,
        trace_id=trace.trace_id,
        metadata={"live_qdt_write": False, "manifest_keys": tuple(sorted(data.keys())) if data else tuple()},
    )


def consolidate_hgm_release_manifests(root_path: str | Path | None = None, config=None, options: Optional[HGM10ReleaseOptions | Mapping[str, Any]] = None) -> ReleaseConsolidationRecord:
    """Collect HGM release manifests and documentation paths into one record."""

    opts = coerce_hgm10_options(options)
    validation = ValidationResult()
    root = Path(root_path or Path.cwd()).resolve()
    release_root = root / "release"
    docs_root = root / "docs" / "hgm_hpme"
    traces = []

    manifest_paths = []
    if release_root.exists():
        manifest_paths = sorted(release_root.glob("hgm_*/manifest.json"))
    if not manifest_paths:
        validation.warning("hgm10_release.no_manifests", "No HGM manifest files found", str(release_root))
    if len(manifest_paths) > opts.max_manifests:
        validation.warning("hgm10_release.manifest_bound", "Manifest list truncated by max_manifests", "release")
    selected_paths = manifest_paths[: opts.max_manifests]
    summaries = tuple(_summarize_manifest(path, root, validation) for path in selected_paths)

    expected_stages = {"hgm_0a", "hgm_0b", "hgm_1", "hgm_2", "hgm_3", "hgm_4", "hgm_5", "hgm_6", "hgm_7", "hgm_8", "hgm_9"}
    found_stages = {summary.stage_id for summary in summaries if summary.found}
    missing = tuple(sorted(expected_stages - found_stages))
    if missing:
        validation.warning("hgm10_release.expected_stage_manifest_missing", f"Expected stage manifests missing: {missing}", "release")

    docs = []
    if docs_root.exists():
        docs = sorted(str(path.relative_to(root)) for path in docs_root.glob("*.md"))
    if not docs:
        validation.warning("hgm10_release.no_docs", "No HGM documentation files found", str(docs_root))

    trace = trace_hgm10("release_consolidation.consolidate_hgm_release_manifests", validation, {"manifest_count": len(summaries), "doc_count": len(docs)})
    traces.append(trace)
    return ReleaseConsolidationRecord(
        consolidation_id=f"hgm10_release_consolidation_{hgm10_stable_hash(opts.release_version, len(summaries), len(docs), missing)}",
        release_version=opts.release_version,
        manifest_summaries=summaries,
        documentation_files=tuple(docs),
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "root_path": str(root),
            "expected_stage_count": len(expected_stages),
            "found_stage_count": len(found_stages),
            "missing_stage_manifests": missing,
            "live_qdt_write": False,
            "production_enabled": False,
        },
    )
