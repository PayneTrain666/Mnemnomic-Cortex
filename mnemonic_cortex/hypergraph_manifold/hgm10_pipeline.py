"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm10 pipeline.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
High-level HGM-10 final release consolidation entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .api_freeze import coerce_hgm10_options, freeze_hgm_public_api
from .hgm10_result import HGM10ReleaseOptions, HGM10ReleaseConsolidationResult, trace_hgm10
from .integration_roadmap import build_hgm_integration_roadmap
from .release_consolidation import consolidate_hgm_release_manifests
from .validation import ValidationResult


def build_hgm10_release_consolidation(root_path: str | Path | None = None, config=None, options: Optional[HGM10ReleaseOptions | Mapping[str, Any]] = None) -> HGM10ReleaseConsolidationResult:
    """Build the HGM v0.1 API freeze, release consolidation, and roadmap."""

    opts = coerce_hgm10_options(options)
    validation = ValidationResult()
    traces = []
    api = freeze_hgm_public_api(config=config, options=opts)
    release = consolidate_hgm_release_manifests(root_path=root_path, config=config, options=opts)
    roadmap = build_hgm_integration_roadmap(config=config, options=opts)
    validation.merge(api.validation).merge(release.validation).merge(roadmap.validation)
    traces.extend(api.trace_records + release.trace_records + roadmap.trace_records)
    trace = trace_hgm10(
        "hgm10_pipeline.build_hgm10_release_consolidation",
        validation,
        {"api_frozen": api.frozen, "manifest_count": len(release.manifest_summaries), "roadmap_items": len(roadmap.items)},
    )
    traces.append(trace)
    return HGM10ReleaseConsolidationResult(
        api_freeze=api,
        release_consolidation=release,
        integration_roadmap=roadmap,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "release_version": opts.release_version,
            "additive_only": True,
            "api_symbols": len(api.symbols),
            "manifest_summaries": len(release.manifest_summaries),
            "roadmap_items": len(roadmap.items),
            "live_qdt_write": False,
            "production_enabled": False,
        },
    )
