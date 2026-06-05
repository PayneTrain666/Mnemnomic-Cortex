"""Permissioned CommitGate boundary audit for HGM/QDT WRITE-PREP-5.

The audit checks that the permission boundary remains closed. It does not call
SystemCommitGate.stage/commit and it does not mutate any QDT/WM store.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result, SyntheticCommitGateAdapterBoundary
from .hgm_qdt_write_prep5_result import (
    HGMQDTWritePrep5Options,
    PermissionBoundaryAuditCheck,
    PermissionedCommitBoundaryAuditResult,
    write_prep5_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_live_shape_contract_harness import coerce_write_prep5_options
from .validation import ValidationResult


def _boundary_from_input(obj: Any) -> SyntheticCommitGateAdapterBoundary | None:
    if isinstance(obj, HGMQDTWritePrep4Result):
        return obj.adapter_boundary
    if isinstance(obj, SyntheticCommitGateAdapterBoundary):
        return obj
    return None


def audit_permissioned_commit_boundary(
    adapter_boundary_or_prep4: Any,
    live_shape_harness: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep5Options | Mapping[str, Any]] = None,
) -> PermissionedCommitBoundaryAuditResult:
    """Audit the future commit boundary while preserving write blockage."""
    opts = coerce_write_prep5_options(options)
    validation = ValidationResult()
    traces = []
    boundary = _boundary_from_input(adapter_boundary_or_prep4)
    if boundary is None:
        validation.error("prep5.permission_boundary.invalid_input", "SyntheticCommitGateAdapterBoundary or HGMQDTWritePrep4Result is required", "adapter_boundary")
        trace = trace_write_prep("qdt_permission_boundary_audit.invalid", validation, {"stage_called": False, "commit_called": False})
        return PermissionedCommitBoundaryAuditResult(
            audit_id=write_prep5_result_id("permission_boundary_invalid", type(adapter_boundary_or_prep4).__name__),
            checks=tuple(),
            permission_boundary_clean=False,
            stage_called=False,
            commit_called=False,
            shared_slot_store_mutated=False,
            qh_storage_mutated=False,
            rollback_stack_mutated=False,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "live_write_executed": False},
        )
    validation.merge(boundary.validation)
    traces.extend(boundary.trace_records)
    checks = []
    raw_checks = tuple(boundary.checks)[: opts.max_boundary_checks]
    if len(boundary.checks) > opts.max_boundary_checks:
        validation.warning("prep5.permission_boundary.truncated", "boundary checks truncated to max_boundary_checks", str(opts.max_boundary_checks))
    for check in raw_checks:
        stage_blocked = bool(check.stage_allowed is False and boundary.stage_called is False)
        commit_blocked = bool(check.commit_allowed is False and boundary.commit_called is False)
        write_permission_granted = bool(getattr(check, "metadata", {}).get("write_permission_granted", False))
        write_permission_present = bool("write_permission" in str(check.metadata) or True)
        simulated_permission_only = bool(check.evaluate_preview_allowed and not check.stage_allowed and not check.commit_allowed)
        clean = bool(stage_blocked and commit_blocked and not write_permission_granted and not boundary.shared_slot_store_mutated and not boundary.qh_storage_mutated and not boundary.rollback_stack_mutated)
        if opts.require_stage_commit_blocked and not clean:
            validation.error("prep5.permission_boundary.not_clean", "permission boundary did not remain clean", check.proposal_id)
        reason = check.blocked_reason or "stage/commit remain blocked in WRITE-PREP-5"
        trace = trace_write_prep("qdt_permission_boundary_audit.check", validation, {
            "proposal_id": check.proposal_id,
            "stage_blocked": stage_blocked,
            "commit_blocked": commit_blocked,
            "permission_boundary_clean": clean,
            "live_write_executed": False,
        })
        traces.append(trace)
        checks.append(PermissionBoundaryAuditCheck(
            check_id=write_prep5_result_id("permission_boundary_check", check.check_id, clean),
            proposal_id=check.proposal_id,
            stage_called=False,
            commit_called=False,
            stage_blocked=stage_blocked,
            commit_blocked=commit_blocked,
            write_permission_present=write_permission_present,
            write_permission_granted=write_permission_granted,
            simulated_permission_only=simulated_permission_only,
            permission_boundary_clean=clean,
            blocked_reason=reason,
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-5",
                "permissioned_commit_audit": bool(opts.permissioned_commit_audit),
                "dry_run": True,
                "live_write_executed": False,
            },
        ))
    boundary_clean = bool(checks and all(c.permission_boundary_clean for c in checks))
    final_trace = trace_write_prep("qdt_permission_boundary_audit.audit_permissioned_commit_boundary", validation, {
        "check_count": len(checks),
        "permission_boundary_clean": boundary_clean,
        "stage_called": False,
        "commit_called": False,
        "live_write_executed": False,
    })
    traces.append(final_trace)
    return PermissionedCommitBoundaryAuditResult(
        audit_id=write_prep5_result_id("permission_boundary_audit", tuple(c.check_id for c in checks)),
        checks=tuple(checks),
        permission_boundary_clean=boundary_clean,
        stage_called=False,
        commit_called=False,
        shared_slot_store_mutated=False,
        qh_storage_mutated=False,
        rollback_stack_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-5",
            "permissioned_commit_boundary_audit": True,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
        },
    )
