"""
Plain-language summary
----------------------
What this file is for: Dry-run or sandbox helper (qdt_shadow_commit_sandbox).
How it fits in the system: Lets engineers rehearse a path safely without committing live side effects.
Status: LOW-USE / SAFETY SCAFFOLD
Important notes for non-coders: Not the everyday training path.

Technical notes (original):
Shadow commit sandbox for WRITE-PREP-7.

The sandbox records what a future commit would need to do, but all stage/commit
and storage mutation flags remain false for live systems.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep6_result import HGMQDTWritePrep6Result, QHStorageRecordSandboxRecord
from .hgm_qdt_write_prep7_result import (
    HGMQDTWritePrep7Options,
    PermissionTokenContractResult,
    ShadowCommitSandboxOperation,
    ShadowCommitSandboxResult,
    write_prep7_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_permission_token_contract import coerce_write_prep7_options
from .validation import ValidationResult


def _qh_records(obj: Any) -> tuple[QHStorageRecordSandboxRecord, ...]:
    if isinstance(obj, HGMQDTWritePrep6Result):
        return tuple(obj.qh_sandbox.records or tuple())
    if isinstance(obj, QHStorageRecordSandboxRecord):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, QHStorageRecordSandboxRecord))
    return tuple()


def build_shadow_commit_sandbox(
    prep6_or_qh_records: Any,
    permission_contract: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep7Options | Mapping[str, Any]] = None,
) -> ShadowCommitSandboxResult:
    """Build shadow commit operations without live stage/commit/storage writes."""
    opts = coerce_write_prep7_options(options)
    validation = ValidationResult()
    traces = []
    qh_records = _qh_records(prep6_or_qh_records)
    token_records = tuple(permission_contract.token_records) if isinstance(permission_contract, PermissionTokenContractResult) else tuple()
    token_by_operation = {r.operation_id: r for r in token_records}
    if not qh_records:
        validation.error("prep7.shadow_commit.empty_qh_records", "QH sandbox records are required", "prep6_or_qh_records")
    if permission_contract is not None and not isinstance(permission_contract, PermissionTokenContractResult):
        validation.warning("prep7.shadow_commit.invalid_permission_contract", "permission_contract is not PermissionTokenContractResult", "permission_contract")
    if len(qh_records) > opts.max_shadow_operations:
        validation.warning("prep7.shadow_commit.bounded_records", "QH record count exceeded max_shadow_operations; records truncated", "qh_records")
    operations: list[ShadowCommitSandboxOperation] = []
    for rec in sorted(qh_records[: opts.max_shadow_operations], key=lambda r: (r.canonical_slot_id, r.proposal_id, r.sandbox_record_id)):
        blockers: list[str] = []
        token = token_by_operation.get(rec.proposal_id) or next((t for t in token_records if t.target_slot_id == rec.canonical_slot_id), None)
        token_id = token.token_contract_id if token is not None else ""
        if token is None:
            blockers.append("permission token contract is missing for operation")
        elif not token.can_authorize_live_write:
            blockers.append("permission token contract does not authorize live write")
        if not rec.validated:
            blockers.append("QH sandbox record is not validated")
        if not rec.canonical_slot_id.startswith("css-"):
            blockers.append("canonical css-* slot binding is missing")
        if rec.write_permission_granted:
            blockers.append("QH sandbox record unexpectedly granted write permission")
        shadow_stage = bool(opts.allow_shadow_commit_sandbox and rec.validated and rec.canonical_slot_id.startswith("css-"))
        shadow_commit = bool(shadow_stage and rec.qh_record_id.startswith("qhrec-"))
        # Shadow success means the isolated shadow path was internally coherent,
        # not that production writes are authorized.
        shadow_success = bool(shadow_stage and shadow_commit and not rec.write_permission_granted)
        if not opts.allow_shadow_commit_sandbox:
            blockers.append("shadow commit sandbox disabled by options")
            shadow_success = False
        trace = trace_write_prep("qdt_shadow_commit_sandbox.operation", validation, {
            "proposal_id": rec.proposal_id,
            "target_slot_id": rec.canonical_slot_id,
            "shadow_stage_simulated": shadow_stage,
            "shadow_commit_simulated": shadow_commit,
            "live_stage_called": False,
            "live_commit_called": False,
            "live_store_mutated": False,
            "live_qh_mutated": False,
            "rollback_stack_mutated": False,
            "shadow_success": shadow_success,
            "blockers": tuple(blockers),
        })
        traces.append(trace)
        operations.append(ShadowCommitSandboxOperation(
            shadow_operation_id=write_prep7_result_id("shadow_commit_op", rec.sandbox_record_id, rec.canonical_slot_id),
            source_token_contract_id=token_id,
            operation_id=rec.proposal_id,
            target_slot_id=rec.canonical_slot_id,
            qh_record_id=rec.qh_record_id,
            rollback_snapshot_ref=f"shadow-snapshot-{rec.canonical_slot_id}-{rec.proposal_id}",
            would_stage=shadow_stage,
            would_commit=shadow_commit,
            shadow_stage_simulated=shadow_stage,
            shadow_commit_simulated=shadow_commit,
            live_stage_called=False,
            live_commit_called=False,
            live_store_mutated=False,
            live_qh_mutated=False,
            rollback_stack_mutated=False,
            shadow_success=shadow_success,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-7",
                "shadow_only": True,
                "production_authorized": False,
            },
        ))
    if not operations:
        validation.warning("prep7.shadow_commit.no_operations", "no shadow commit operations were produced", "operations")
    trace = trace_write_prep("qdt_shadow_commit_sandbox.build_shadow_commit_sandbox", validation, {
        "operation_count": len(operations),
        "shadow_success": bool(operations and all(o.shadow_success for o in operations)),
        "live_stage_called": False,
        "live_commit_called": False,
        "live_store_mutated": False,
        "live_qh_mutated": False,
        "rollback_stack_mutated": False,
    })
    traces.append(trace)
    return ShadowCommitSandboxResult(
        sandbox_id=write_prep7_result_id("shadow_commit_sandbox", tuple(o.shadow_operation_id for o in operations)),
        operations=tuple(operations),
        shadow_success=bool(operations and all(o.shadow_success for o in operations)),
        live_stage_called=False,
        live_commit_called=False,
        live_store_mutated=False,
        live_qh_mutated=False,
        rollback_stack_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-7",
            "shadow_commit_only": True,
            "live_write_executed": False,
            "production_write_ready": False,
        },
    )
