"""Synthetic CommitGate adapter boundary for WRITE-PREP-4.

The boundary evaluates whether real contract-object previews could cross into a
future adapter, while explicitly blocking stage/commit/write methods in this
stage.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import (
    HGMQDTWritePrep4Options,
    RealContractObjectConstructionResult,
    SyntheticCommitGateAdapterBoundary,
    SyntheticCommitGateBoundaryCheck,
    write_prep4_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_real_contract_object_dryrun import coerce_write_prep4_options
from .validation import ValidationResult


def build_synthetic_commitgate_adapter_boundary(
    contract_result: Any,
    config=None,
    options: Optional[HGMQDTWritePrep4Options | Mapping[str, Any]] = None,
) -> SyntheticCommitGateAdapterBoundary:
    """Create a synthetic adapter boundary report without invoking CommitGate."""
    opts = coerce_write_prep4_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(contract_result, RealContractObjectConstructionResult):
        validation.error("prep4.boundary.invalid_contract_result", "RealContractObjectConstructionResult is required", "contract_result")
        trace = trace_write_prep("qdt_synthetic_commitgate_adapter_boundary.invalid", validation, {"stage_called": False, "commit_called": False})
        return SyntheticCommitGateAdapterBoundary(
            boundary_id=write_prep4_result_id("synthetic_adapter_boundary_invalid", type(contract_result).__name__),
            checks=tuple(),
            stage_called=False,
            commit_called=False,
            shared_slot_store_mutated=False,
            qh_storage_mutated=False,
            rollback_stack_mutated=False,
            evaluation_preview_count=0,
            validation=validation,
            trace_records=(trace,),
            metadata={"invalid": True, "synthetic_boundary_only": True},
        )
    validation.merge(contract_result.validation)
    traces.extend(contract_result.trace_records)
    checks: list[SyntheticCommitGateBoundaryCheck] = []
    for preview in contract_result.previews:
        contract_validated = bool(preview.constructed and preview.validation_ready and not preview.write_permission)
        blocked = "stage/commit disabled in WRITE-PREP-4 synthetic adapter boundary"
        if not contract_validated:
            blocked = "; ".join(preview.blockers or ("contract preview not validated",))
        trace = trace_write_prep("qdt_synthetic_commitgate_adapter_boundary.check", validation, {
            "proposal_preview_id": preview.preview_id,
            "proposal_id": preview.proposal_id,
            "contract_validated": contract_validated,
            "stage_allowed": False,
            "commit_allowed": False,
            "evaluate_preview_allowed": opts.allow_adapter_evaluate_preview,
        })
        traces.append(trace)
        checks.append(SyntheticCommitGateBoundaryCheck(
            check_id=write_prep4_result_id("synthetic_boundary_check", preview.preview_id, contract_validated),
            proposal_preview_id=preview.preview_id,
            proposal_id=preview.proposal_id,
            stage_allowed=False,
            commit_allowed=False,
            evaluate_preview_allowed=bool(opts.allow_adapter_evaluate_preview and contract_validated),
            contract_validated=contract_validated,
            blocked_reason=blocked,
            trace_id=trace.trace_id,
            metadata={
                "synthetic_adapter_boundary": True,
                "system_commitgate_stage_called": False,
                "system_commitgate_commit_called": False,
                "live_write_executed": False,
            },
        ))
    final_trace = trace_write_prep("qdt_synthetic_commitgate_adapter_boundary.build_synthetic_commitgate_adapter_boundary", validation, {
        "check_count": len(checks),
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return SyntheticCommitGateAdapterBoundary(
        boundary_id=write_prep4_result_id("synthetic_adapter_boundary", tuple(c.check_id for c in checks)),
        checks=tuple(checks),
        stage_called=False,
        commit_called=False,
        shared_slot_store_mutated=False,
        qh_storage_mutated=False,
        rollback_stack_mutated=False,
        evaluation_preview_count=sum(1 for c in checks if c.evaluate_preview_allowed),
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-4",
            "synthetic_adapter_boundary": True,
            "no_system_commitgate_stage": True,
            "no_system_commitgate_commit": True,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
        },
    )
