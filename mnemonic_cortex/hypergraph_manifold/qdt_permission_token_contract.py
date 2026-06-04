"""Permission-token contract builder for WRITE-PREP-7.

This module defines what a future production write permission token must prove.
It does not create a real token and does not authorize live writes.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep6_result import HGMQDTWritePrep6Result, RollbackSnapshotBindingDryRunRecord
from .hgm_qdt_write_prep7_result import (
    HGMQDTWritePrep7Options,
    PermissionTokenContractRecord,
    PermissionTokenContractResult,
    write_prep7_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .validation import ValidationResult


def coerce_write_prep7_options(options: Optional[HGMQDTWritePrep7Options | Mapping[str, Any]] = None) -> HGMQDTWritePrep7Options:
    if options is None:
        return HGMQDTWritePrep7Options()
    if isinstance(options, HGMQDTWritePrep7Options):
        return options
    if isinstance(options, Mapping):
        allowed = {k: v for k, v in dict(options).items() if k in HGMQDTWritePrep7Options.__dataclass_fields__}
        return HGMQDTWritePrep7Options(**allowed)
    return HGMQDTWritePrep7Options()


def _rollback_bindings(obj: Any) -> tuple[RollbackSnapshotBindingDryRunRecord, ...]:
    if isinstance(obj, HGMQDTWritePrep6Result):
        return tuple(obj.rollback_binding_dry_run.bindings or tuple())
    if isinstance(obj, RollbackSnapshotBindingDryRunRecord):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, RollbackSnapshotBindingDryRunRecord))
    return tuple()


def build_permission_token_contract(
    prep6_or_bindings: Any,
    config=None,
    options: Optional[HGMQDTWritePrep7Options | Mapping[str, Any]] = None,
) -> PermissionTokenContractResult:
    """Build future permission-token contract records without granting writes."""
    opts = coerce_write_prep7_options(options)
    validation = ValidationResult()
    traces = []
    bindings = _rollback_bindings(prep6_or_bindings)
    if not bindings:
        validation.error("prep7.permission_token.empty_bindings", "rollback binding dry-run records are required", "prep6_or_bindings")
    if len(bindings) > opts.max_permission_tokens:
        validation.warning("prep7.permission_token.bounded_records", "binding count exceeded max_permission_tokens; records truncated", "bindings")
    records: list[PermissionTokenContractRecord] = []
    for binding in sorted(bindings[: opts.max_permission_tokens], key=lambda b: (b.target_slot_id, b.operation_id, b.binding_id)):
        blockers: list[str] = []
        token_present = False
        token_validated = False
        human_marker = False
        write_granted = False
        if opts.require_explicit_permission_token and not token_present:
            blockers.append("explicit permission token is absent")
        if opts.require_human_approval_marker and not human_marker:
            blockers.append("human approval marker is absent")
        if opts.require_rollback_binding_ready and not bool(binding.binding_ready):
            blockers.append("rollback binding is not production-ready")
        if bool(binding.rollback_stack_mutated):
            blockers.append("rollback stack mutation is not allowed during WRITE-PREP-7")
        can_authorize = bool(token_present and token_validated and human_marker and not blockers and write_granted)
        trace = trace_write_prep("qdt_permission_token_contract.record", validation, {
            "operation_id": binding.operation_id,
            "target_slot_id": binding.target_slot_id,
            "token_present": token_present,
            "token_validated": token_validated,
            "human_approval_marker_present": human_marker,
            "write_permission_granted": write_granted,
            "can_authorize_live_write": can_authorize,
            "blockers": tuple(blockers),
        })
        traces.append(trace)
        records.append(PermissionTokenContractRecord(
            token_contract_id=write_prep7_result_id("perm_token_contract", binding.binding_id, binding.operation_id, binding.target_slot_id),
            source_binding_id=binding.binding_id,
            operation_id=binding.operation_id,
            target_slot_id=binding.target_slot_id,
            required_scope="hgm_qdt_write_execute",
            permission_token_id_preview=f"ptok-{write_prep_stable_hash(binding.operation_id, binding.target_slot_id)}",
            token_present=token_present,
            token_validated=token_validated,
            human_approval_marker_present=human_marker,
            write_permission_granted=write_granted,
            can_authorize_live_write=can_authorize,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-7",
                "contract_only": True,
                "required_evidence": (
                    "explicit_permission_token",
                    "human_approval_marker",
                    "rollback_snapshot_binding_ready",
                    "production_write_stage_approval",
                ),
            },
        ))
    if not records:
        validation.warning("prep7.permission_token.no_records", "no permission-token contract records were produced", "records")
    token_ready = bool(records and all(r.can_authorize_live_write for r in records))
    trace = trace_write_prep("qdt_permission_token_contract.build_permission_token_contract", validation, {
        "record_count": len(records),
        "token_contract_ready": token_ready,
        "live_write_authorized": False,
    })
    traces.append(trace)
    return PermissionTokenContractResult(
        contract_id=write_prep7_result_id("permission_token_contract_result", tuple(r.token_contract_id for r in records)),
        token_records=tuple(records),
        token_contract_ready=token_ready,
        live_write_authorized=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-7",
            "dry_run": True,
            "live_write_authorized": False,
            "permission_contract_only": True,
        },
    )
