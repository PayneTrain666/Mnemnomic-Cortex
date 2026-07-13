"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt commitgate preflight.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
CommitGate-style preflight checks for HGM/QDT dry-run proposals.

This module deliberately avoids SystemCommitGate.stage and commit. It evaluates
proposal preview records against the gates expected by QDT/WM contracts.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep_result import HGMQDTWritePrepResult, trace_write_prep, write_prep_stable_hash
from .hgm_qdt_write_prep2_result import (
    CommitGatePreflightCheck,
    CommitGatePreflightResult,
    DryRunProposalBuilderResult,
    DryRunSystemWriteProposalPreview,
    HGMQDTWritePrep2Options,
)
from .qdt_dry_run_proposal_builder import _coerce_options, build_dry_run_system_write_proposal_previews
from .validation import ValidationResult


def _add_check(checks: list[CommitGatePreflightCheck], traces: list, validation: ValidationResult, proposal_id: str, name: str, passed: bool, blocking: bool, reason: str) -> None:
    if blocking and not passed:
        validation.error(f"preflight.{name}.failed", reason, proposal_id)
    elif not passed:
        validation.warning(f"preflight.{name}.warning", reason, proposal_id)
    trace = trace_write_prep("qdt_commitgate_preflight.check", validation, {"proposal_id": proposal_id, "check": name, "passed": passed, "blocking": blocking})
    traces.append(trace)
    checks.append(CommitGatePreflightCheck(
        check_id=f"preflight_{write_prep_stable_hash(proposal_id, name)}",
        proposal_id=proposal_id,
        check_name=name,
        passed=bool(passed),
        blocking=bool(blocking),
        reason=reason,
        trace_id=trace.trace_id,
        metadata={"stage_called": False, "commit_called": False, "live_writes": False},
    ))


def _proposal_previews_from_input(proposals_or_builder: Any) -> tuple[DryRunSystemWriteProposalPreview, ...]:
    if isinstance(proposals_or_builder, DryRunProposalBuilderResult):
        return tuple(proposals_or_builder.proposals or tuple())
    if isinstance(proposals_or_builder, DryRunSystemWriteProposalPreview):
        return (proposals_or_builder,)
    if isinstance(proposals_or_builder, HGMQDTWritePrepResult):
        return tuple(build_dry_run_system_write_proposal_previews(proposals_or_builder).proposals)
    if isinstance(proposals_or_builder, (list, tuple)):
        return tuple(p for p in proposals_or_builder if isinstance(p, DryRunSystemWriteProposalPreview))
    return tuple()


def run_commitgate_preflight(proposals_or_builder: Any, write_prep_result: HGMQDTWritePrepResult | None = None, config=None, options: Optional[HGMQDTWritePrep2Options | Mapping[str, Any]] = None) -> CommitGatePreflightResult:
    """Run non-mutating CommitGate-style preflight checks."""
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    proposals = _proposal_previews_from_input(proposals_or_builder)
    if not proposals:
        validation.warning("preflight.empty_proposals", "no proposal previews supplied", "proposals")
    checks: list[CommitGatePreflightCheck] = []
    rollback_ready = False
    qh_ready_global = True
    slot_ready_global = True
    if isinstance(write_prep_result, HGMQDTWritePrepResult):
        rollback_ready = bool(write_prep_result.rollback_handshake.complete)
        qh_ready_global = bool(write_prep_result.qspin_qh_contract.conversions)
        slot_ready_global = bool(write_prep_result.slot_mapping_plan.mappings)
    for prop in sorted(proposals[: opts.max_proposals], key=lambda p: (p.local_slot_id, p.proposal_id)):
        _add_check(checks, traces, validation, prop.proposal_id, "proposal_ready", prop.ready, True, "proposal preview is internally ready" if prop.ready else "; ".join(prop.blockers or ("proposal preview not ready",)))
        _add_check(checks, traces, validation, prop.proposal_id, "content_shape", prop.content_shape == (len(prop.content_vector),) and len(prop.content_vector) <= opts.max_tensor_dim, True, "content shape is valid and bounded")
        _add_check(checks, traces, validation, prop.proposal_id, "finite_content", all(math.isfinite(v) for v in prop.content_vector), True, "content vector is finite")
        _add_check(checks, traces, validation, prop.proposal_id, "write_permission_false", prop.write_permission is False, True, "write_permission is false by default")
        _add_check(checks, traces, validation, prop.proposal_id, "slot_mapping", bool(prop.local_slot_id and prop.canonical_slot_id and slot_ready_global), opts.require_slot_mapping, "slot mapping is present")
        _add_check(checks, traces, validation, prop.proposal_id, "qh_conversion", bool(prop.geometry_map and qh_ready_global), opts.require_qh_conversion, "QH conversion preview is present")
        _add_check(checks, traces, validation, prop.proposal_id, "rollback_handshake", rollback_ready, opts.require_rollback_handshake, "rollback handshake is bound to actual snapshots" if rollback_ready else "rollback handshake is not bound to actual snapshots")
        _add_check(checks, traces, validation, prop.proposal_id, "no_stage_or_commit", prop.metadata.get("stage_called") is False and prop.metadata.get("commit_called") is False, True, "no stage/commit call occurred")
    blocking_failures = [c for c in checks if c.blocking and not c.passed]
    readiness_score = 0.0 if not checks else sum(1.0 for c in checks if c.passed) / float(len(checks))
    ready = bool(checks and not blocking_failures)
    final_trace = trace_write_prep("qdt_commitgate_preflight.run_commitgate_preflight", validation, {
        "proposal_count": len(proposals),
        "check_count": len(checks),
        "ready": ready,
        "readiness_score": readiness_score,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return CommitGatePreflightResult(
        checks=tuple(checks),
        ready=ready,
        readiness_score=readiness_score,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "dry_run": True,
            "live_writes": False,
            "stage_called": False,
            "commit_called": False,
            "blocking_failures": tuple(c.check_id for c in blocking_failures),
        },
    )
