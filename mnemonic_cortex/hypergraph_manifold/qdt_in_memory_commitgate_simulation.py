"""Isolated in-memory CommitGate simulation for HGM/QDT WRITE-PREP-3.

This module simulates staging/commit behavior against immutable preview records
and a synthetic slot-store sandbox. It never calls SystemCommitGate.stage,
SystemCommitGate.commit, SharedSlotStore writes, QH storage writes, or rollback
stack mutation.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .hgm_qdt_write_prep2_result import (
    CommitGatePreflightResult,
    DryRunProposalBuilderResult,
    DryRunSystemWriteProposalPreview,
    EndToEndWriteSimulationReport,
    HGMQDTWritePrep2Result,
)
from .hgm_qdt_write_prep3_result import (
    HGMQDTWritePrep3Options,
    InMemoryCommitGateSimulationOperation,
    InMemoryCommitGateSimulationResult,
    write_prep3_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_synthetic_slot_store import (
    apply_synthetic_slot_writes,
    build_synthetic_shared_slot_store_sandbox,
    coerce_write_prep3_options,
)
from .validation import ValidationResult


def _extract_prep2_parts(obj: Any) -> tuple[tuple[DryRunSystemWriteProposalPreview, ...], CommitGatePreflightResult | None, Any | None]:
    if isinstance(obj, HGMQDTWritePrep2Result):
        return tuple(obj.proposal_builder_result.proposals), obj.preflight_result, obj.simulation_report
    if isinstance(obj, EndToEndWriteSimulationReport):
        return tuple(obj.proposal_builder_result.proposals), obj.preflight_result, obj
    if isinstance(obj, DryRunProposalBuilderResult):
        return tuple(obj.proposals), None, None
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, DryRunSystemWriteProposalPreview)), None, None
    return tuple(), None, None


def _preflight_pass_by_proposal(preflight: CommitGatePreflightResult | None) -> dict[str, bool]:
    if preflight is None:
        return {}
    out: dict[str, bool] = {}
    by_id: dict[str, list] = {}
    for check in preflight.checks:
        by_id.setdefault(check.proposal_id, []).append(check)
    for proposal_id, checks in by_id.items():
        out[proposal_id] = not any(c.blocking and not c.passed for c in checks)
    return out


def simulate_in_memory_commitgate(
    prep2_result_or_proposals: Any,
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> InMemoryCommitGateSimulationResult:
    """Run an isolated in-memory CommitGate simulation over proposal previews."""
    opts = coerce_write_prep3_options(options)
    validation = ValidationResult()
    traces = []
    proposals, preflight, report = _extract_prep2_parts(prep2_result_or_proposals)
    if not proposals:
        validation.error("prep3.commitgate.empty_proposals", "no dry-run proposal previews supplied", "prep2_result_or_proposals")
    if len(proposals) > opts.max_operations:
        validation.warning("prep3.commitgate.bounded_operation_count", "proposal count exceeded max_operations; simulation truncated", "proposals")
    proposals = tuple(sorted(proposals[: opts.max_operations], key=lambda p: (p.local_slot_id, p.proposal_id)))
    sandbox_before = build_synthetic_shared_slot_store_sandbox(proposals, config=config, options=opts)
    validation.merge(sandbox_before.validation)
    traces.extend(sandbox_before.trace_records)
    preflight_ready = bool(preflight.ready) if preflight is not None else False
    preflight_by_proposal = _preflight_pass_by_proposal(preflight)
    committed_ids: list[str] = []
    operations: list[InMemoryCommitGateSimulationOperation] = []
    slot_before_fp = {slot.slot_id: slot.current_fingerprint for slot in sandbox_before.slots}
    for proposal in proposals:
        blocked = ""
        staged = False
        committed = False
        if opts.require_proposal_ready and not proposal.ready:
            blocked = "; ".join(proposal.blockers or ("proposal preview not ready",))
        elif not opts.allow_synthetic_commit:
            blocked = "synthetic commit disabled by options"
        elif preflight is not None and not preflight_by_proposal.get(proposal.proposal_id, False) and not opts.allow_preflight_blocked_simulation:
            blocked = "preflight blocked proposal and allow_preflight_blocked_simulation=False"
        else:
            staged = True
            committed = True
            committed_ids.append(proposal.proposal_id)
        trace = trace_write_prep("qdt_in_memory_commitgate_simulation.operation", validation, {
            "proposal_id": proposal.proposal_id,
            "staged": staged,
            "committed": committed,
            "blocked_reason": blocked,
            "synthetic_only": True,
            "live_write_executed": False,
        })
        traces.append(trace)
        after_fp = proposal.content_fingerprint if committed else slot_before_fp.get(proposal.local_slot_id, "")
        operations.append(InMemoryCommitGateSimulationOperation(
            operation_id=write_prep3_result_id("synthetic_commit_op", proposal.proposal_id, proposal.local_slot_id),
            proposal_id=proposal.proposal_id,
            target_slot_id=proposal.local_slot_id,
            canonical_slot_id=proposal.canonical_slot_id,
            staged=staged,
            committed=committed,
            blocked_reason=blocked,
            before_fingerprint=slot_before_fp.get(proposal.local_slot_id, ""),
            after_fingerprint=after_fp,
            trace_id=trace.trace_id,
            metadata={
                "synthetic_only": True,
                "preflight_ready": preflight_ready,
                "preflight_passed_for_proposal": preflight_by_proposal.get(proposal.proposal_id),
                "system_commitgate_stage_called": False,
                "system_commitgate_commit_called": False,
                "live_write_executed": False,
            },
        ))
    sandbox_after = apply_synthetic_slot_writes(sandbox_before, proposals, committed_ids, config=config, options=opts)
    validation.merge(sandbox_after.validation)
    traces.extend(sandbox_after.trace_records)
    synthetic_store_mutated = bool(committed_ids)
    final_trace = trace_write_prep("qdt_in_memory_commitgate_simulation.simulate_in_memory_commitgate", validation, {
        "operation_count": len(operations),
        "committed_count": len(committed_ids),
        "synthetic_store_mutated": synthetic_store_mutated,
        "live_store_mutated": False,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return InMemoryCommitGateSimulationResult(
        simulation_id=write_prep3_result_id("in_memory_commitgate_sim", tuple(op.operation_id for op in operations), synthetic_store_mutated),
        operations=tuple(operations),
        sandbox_before=sandbox_before,
        sandbox_after=sandbox_after,
        preflight_ready=preflight_ready,
        synthetic_store_mutated=synthetic_store_mutated,
        live_store_mutated=False,
        stage_called=False,
        commit_called=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-3",
            "isolated_in_memory": True,
            "synthetic_shared_slot_store": True,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "live_qdt_wm_mutated": False,
            "source_report_id": getattr(report, "report_id", ""),
        },
    )
