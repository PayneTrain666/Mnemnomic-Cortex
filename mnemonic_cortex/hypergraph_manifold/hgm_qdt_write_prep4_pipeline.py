"""High-level HGM/QDT WRITE-PREP-4 pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep_result import HGMQDTWritePrepResult, trace_write_prep, write_prep_stable_hash
from .hgm_qdt_write_prep2_result import HGMQDTWritePrep2Result
from .hgm_qdt_write_prep3_result import HGMQDTWritePrep3Result
from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Options, HGMQDTWritePrep4Result
from .qdt_real_contract_object_dryrun import build_real_contract_object_previews, coerce_write_prep4_options
from .qdt_synthetic_commitgate_adapter_boundary import build_synthetic_commitgate_adapter_boundary
from .qdt_rollback_snapshot_binding_plan import build_rollback_snapshot_binding_plan
from .validation import ValidationResult


def build_hgm_qdt_write_prep_4(
    prep2_result: Any,
    prep1_result: Any = None,
    prep3_result: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep4Options | Mapping[str, Any]] = None,
) -> HGMQDTWritePrep4Result:
    """Build WRITE-PREP-4 dry-run contract-object/boundary/binding result.

    ``prep2_result`` supplies dry-run SystemWriteProposal previews. ``prep1_result``
    supplies the rollback handshake contract. ``prep3_result`` optionally supplies
    synthetic rollback replay evidence. No live QDT/WM state is written.
    """
    opts = coerce_write_prep4_options(options)
    validation = ValidationResult()
    traces = []
    contract_objects = build_real_contract_object_previews(prep2_result, config=config, options=opts)
    validation.merge(contract_objects.validation)
    traces.extend(contract_objects.trace_records)
    boundary = build_synthetic_commitgate_adapter_boundary(contract_objects, config=config, options=opts)
    validation.merge(boundary.validation)
    traces.extend(boundary.trace_records)
    if isinstance(prep1_result, HGMQDTWritePrepResult):
        handshake = prep1_result.rollback_handshake
    else:
        handshake = getattr(prep1_result, "rollback_handshake", None)
    rollback_plan = build_rollback_snapshot_binding_plan(handshake, rollback_replay=prep3_result, config=config, options=opts)
    validation.merge(rollback_plan.validation)
    traces.extend(rollback_plan.trace_records)
    final_trace = trace_write_prep("hgm_qdt_write_prep4_pipeline.build_hgm_qdt_write_prep_4", validation, {
        "contract_preview_count": len(contract_objects.previews),
        "adapter_boundary_checks": len(boundary.checks),
        "rollback_bindings": len(rollback_plan.bindings),
        "live_write_executed": False,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return HGMQDTWritePrep4Result(
        contract_object_result=contract_objects,
        adapter_boundary=boundary,
        rollback_binding_plan=rollback_plan,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-4",
            "dry_run": True,
            "read_only": True,
            "real_contract_object_preview_only": True,
            "synthetic_commitgate_adapter_boundary": True,
            "rollback_snapshot_binding_plan": True,
            "live_write_executed": False,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
            "result_id": f"hgm_qdt_write_prep_4_{write_prep_stable_hash(len(contract_objects.previews), len(rollback_plan.bindings))}",
        },
    )
