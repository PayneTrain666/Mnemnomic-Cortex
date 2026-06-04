"""Integration roadmap generation for HGM v0.1 and later."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .api_freeze import coerce_hgm10_options
from .hgm10_result import HGM10ReleaseOptions, IntegrationRoadmapItem, IntegrationRoadmapRecord, hgm10_stable_hash, trace_hgm10
from .validation import ValidationResult

_NEXT_COMMAND = "DEV-FLOW FINALIZE HGM-V0.1 — Preserve Hypergraph Manifold / Hyperset Probability Matrix Expansion Release State"


def build_hgm_integration_roadmap(config=None, options: Optional[HGM10ReleaseOptions | Mapping[str, Any]] = None) -> IntegrationRoadmapRecord:
    """Build the post-HGM-10 integration roadmap without enabling production execution."""

    opts = coerce_hgm10_options(options)
    validation = ValidationResult()
    traces = []
    raw_items = [
        ("HGM-V0.1-FINALIZE", "Finalize HGM v0.1 release state", "high", "next", ()),
        ("HGM-QDT-BRIDGE-AUDIT", "Audit QDT/WM bridge contracts against live package internals", "high", "planned", ("HGM-V0.1-FINALIZE",)),
        ("HGM-WRITE-PERMISSION-STAGE", "Design explicit write-capable stage with hard approval gate", "high", "planned", ("HGM-QDT-BRIDGE-AUDIT",)),
        ("HGM-LEARNED-EMBEDDINGS", "Replace deterministic scaffold embeddings with trained/evaluated runtime embeddings", "medium", "planned", ("HGM-V0.1-FINALIZE",)),
        ("HGM-ROBOTICS-SIM-BRIDGE", "Connect advisory robotics planning to simulator-only evaluation harness", "medium", "planned", ("HGM-LEARNED-EMBEDDINGS",)),
        ("HGM-PRODUCTION-READINESS", "Run production-readiness gate after write execution and replay verification mature", "high", "blocked", ("HGM-WRITE-PERMISSION-STAGE", "HGM-LEARNED-EMBEDDINGS")),
    ]
    if len(raw_items) > opts.max_roadmap_items:
        validation.warning("hgm10_roadmap.item_bound", "Roadmap item list truncated by max_roadmap_items", "items")
    items = []
    for stage, title, priority, status, blocked_by in raw_items[: opts.max_roadmap_items]:
        trace = trace_hgm10("integration_roadmap.item", validation, {"stage": stage, "title": title})
        traces.append(trace)
        items.append(
            IntegrationRoadmapItem(
                item_id=f"hgm10_roadmap_item_{hgm10_stable_hash(opts.release_version, stage, title)}",
                stage=stage,
                title=title,
                priority=priority,
                status=status,
                blocked_by=tuple(blocked_by),
                trace_id=trace.trace_id,
                metadata={"live_qdt_write": False, "production_enabled": False},
            )
        )
    trace = trace_hgm10("integration_roadmap.build_hgm_integration_roadmap", validation, {"item_count": len(items), "next_command": _NEXT_COMMAND})
    traces.append(trace)
    return IntegrationRoadmapRecord(
        roadmap_id=f"hgm10_roadmap_{hgm10_stable_hash(opts.release_version, tuple(item.stage for item in items))}",
        release_version=opts.release_version,
        items=tuple(items),
        next_command=_NEXT_COMMAND,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"roadmap_type": "post_v0_1", "live_qdt_write": False, "production_enabled": False},
    )
