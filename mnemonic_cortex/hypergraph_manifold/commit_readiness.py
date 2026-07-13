"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: commit readiness.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Commit-readiness scoring for HGM-6 preview-only transactions.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, List, Mapping, Optional

from .hgm4_result import TraceSafeMemoryPlan
from .hgm5_result import HGM5EmbeddingEvaluationResult, IntegrationScoringResult
from .hgm6_result import CommitReadinessScore, HGM6CommitOptions, RollbackManifest, WritePermissionState
from .validation import ValidationResult
from .write_permission_gate import coerce_hgm6_options, trace_hgm6


def _stable_hash(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def _clamp01(value: float) -> float:
    try:
        if not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _extract_hgm5_score(hgm5_result: Any, fallback: float) -> tuple[float, bool]:
    if isinstance(hgm5_result, HGM5EmbeddingEvaluationResult):
        return _clamp01(hgm5_result.integration_scoring_result.aggregate_score), True
    if isinstance(hgm5_result, IntegrationScoringResult):
        return _clamp01(hgm5_result.aggregate_score), True
    if hasattr(hgm5_result, "integration_scoring_result"):
        return _clamp01(getattr(hgm5_result.integration_scoring_result, "aggregate_score", fallback)), True
    if hasattr(hgm5_result, "aggregate_score"):
        return _clamp01(getattr(hgm5_result, "aggregate_score", fallback)), True
    return _clamp01(fallback), False


def score_commit_readiness(
    memory_plan: TraceSafeMemoryPlan,
    hgm5_result=None,
    rollback_manifest: Optional[RollbackManifest] = None,
    write_permission: Optional[WritePermissionState] = None,
    config=None,
    options: Optional[HGM6CommitOptions | Mapping[str, Any]] = None,
) -> CommitReadinessScore:
    """Score preview-only commit readiness without allowing live writes."""

    opts = coerce_hgm6_options(options)
    validation = ValidationResult()
    blockers: List[str] = []
    warnings: List[str] = []
    if not isinstance(memory_plan, TraceSafeMemoryPlan):
        blockers.append("memory plan is missing or invalid")
        validation.error("hgm6_readiness.invalid_plan", blockers[-1], "memory_plan")
        adapter_score = 0.0
        op_score = 0.0
        dry_score = 0.0
    else:
        adapter_score = 1.0 if memory_plan.adapter_status.available else 0.0
        if not memory_plan.adapter_status.available:
            blockers.append("adapter unavailable")
            validation.error("hgm6_readiness.adapter_unavailable", "adapter unavailable blocks commit preview", "adapter_status")
        op_count = len(memory_plan.slot_hooks)
        op_score = 1.0 if op_count > 0 else 0.0
        if op_count == 0:
            warnings.append("no planned operations")
            validation.warning("hgm6_readiness.empty_operations", "no planned operations found", "slot_hooks")
        dry_score = 1.0 if memory_plan.dry_run and not memory_plan.write_intent else 0.0
        if not memory_plan.dry_run or memory_plan.write_intent:
            blockers.append("memory plan is not dry-run safe")
            validation.error("hgm6_readiness.not_dry_run_safe", "memory plan must be dry_run without write_intent", "memory_plan")
    if rollback_manifest is None:
        blockers.append("rollback manifest missing")
        validation.error("hgm6_readiness.rollback_missing", "rollback manifest is required for readiness", "rollback_manifest")
        rollback_score = 0.0
    else:
        rollback_score = 1.0 if rollback_manifest.complete else 0.0
        validation.merge(rollback_manifest.validation)
        if not rollback_manifest.complete:
            blockers.append("rollback manifest incomplete")
            validation.error("hgm6_readiness.rollback_incomplete", "rollback coverage is incomplete", "rollback_manifest")
    hgm5_score, hgm5_present = _extract_hgm5_score(hgm5_result, opts.conservative_missing_hgm5_score)
    if not hgm5_present:
        warnings.append("HGM-5 integration score missing; conservative fallback used")
        validation.warning("hgm6_readiness.hgm5_missing", "using conservative fallback readiness score", "hgm5_result")
    permission_score = 1.0 if (write_permission is not None and write_permission.requested and write_permission.granted) else 0.0
    if write_permission is None or not write_permission.granted:
        blockers.append("write permission not granted")
        validation.warning("hgm6_readiness.permission_denied", "write permission is not granted", "write_permission")
    component_scores = (adapter_score, dry_score, op_score, rollback_score, hgm5_score, permission_score)
    score = _clamp01(sum(component_scores) / len(component_scores))
    ready = bool(not blockers and score >= opts.readiness_threshold)
    if not ready and score >= opts.readiness_threshold and blockers:
        warnings.append("score threshold met but blockers remain")
    trace = trace_hgm6("commit_readiness.score_commit_readiness", validation, {"score": score, "ready": ready, "blockers": tuple(blockers), "warnings": tuple(warnings)})
    return CommitReadinessScore(
        score_id=f"hgm6_ready_{_stable_hash(score, ready, tuple(blockers), hgm5_score)}",
        score=score,
        confidence=0.85 if hgm5_present else 0.55,
        ready=ready,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        trace_id=trace.trace_id,
        metadata={
            "adapter_score": adapter_score,
            "dry_run_score": dry_score,
            "operation_score": op_score,
            "rollback_score": rollback_score,
            "hgm5_score": hgm5_score,
            "permission_score": permission_score,
            "threshold": opts.readiness_threshold,
        },
    )
