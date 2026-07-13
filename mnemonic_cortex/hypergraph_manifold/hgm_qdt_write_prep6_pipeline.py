"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm qdt write prep6 pipeline.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
High-level WRITE-PREP-6 pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result
from .hgm_qdt_write_prep5_result import HGMQDTWritePrep5Result
from .hgm_qdt_write_prep6_result import HGMQDTWritePrep6Options, HGMQDTWritePrep6Result, write_prep6_result_id
from .hgm_qdt_write_prep_result import trace_write_prep
from .qdt_real_shared_slot_store_parity import build_real_shared_slot_store_parity_harness, coerce_write_prep6_options
from .qdt_qh_storage_record_sandbox import build_qh_storage_record_sandbox
from .qdt_rollback_snapshot_binding_dryrun import build_rollback_snapshot_binding_dry_run
from .validation import ValidationResult


def build_hgm_qdt_write_prep_6(
    prep4_result: Any,
    prep5_result: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep6Options | Mapping[str, Any]] = None,
) -> HGMQDTWritePrep6Result:
    """Build WRITE-PREP-6 parity/sandbox/rollback dry-run reports.

    ``prep4_result`` is required because it carries real contract-object previews
    and rollback binding plans. ``prep5_result`` is optional and merged when
    provided to preserve the blocker burn-down lineage.
    """
    opts = coerce_write_prep6_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(prep4_result, HGMQDTWritePrep4Result):
        validation.error("prep6.pipeline.invalid_prep4", "HGMQDTWritePrep4Result is required", "prep4_result")
        shared = build_real_shared_slot_store_parity_harness(prep4_result, options=opts)
        qh = build_qh_storage_record_sandbox(prep4_result, shared, options=opts)
        rollback = build_rollback_snapshot_binding_dry_run(prep4_result, shared, qh, options=opts)
    else:
        validation.merge(prep4_result.validation)
        traces.extend(prep4_result.trace_records)
        if isinstance(prep5_result, HGMQDTWritePrep5Result):
            validation.merge(prep5_result.validation)
            traces.extend(prep5_result.trace_records)
        elif prep5_result is not None:
            validation.warning("prep6.pipeline.invalid_prep5", "prep5_result was supplied but is not HGMQDTWritePrep5Result", "prep5_result")
        shared = build_real_shared_slot_store_parity_harness(prep4_result, options=opts)
        qh = build_qh_storage_record_sandbox(prep4_result, shared, options=opts)
        rollback = build_rollback_snapshot_binding_dry_run(prep4_result, shared, qh, options=opts)
    validation.merge(shared.validation).merge(qh.validation).merge(rollback.validation)
    traces.extend(shared.trace_records)
    traces.extend(qh.trace_records)
    traces.extend(rollback.trace_records)
    trace = trace_write_prep("hgm_qdt_write_prep6_pipeline.build_hgm_qdt_write_prep_6", validation, {
        "shared_slot_parity_ok": shared.parity_ok,
        "qh_validated_count": qh.validated_count,
        "rollback_binding_ready": rollback.binding_ready,
        "live_write_executed": False,
        "live_store_mutated": False,
        "live_qh_storage_mutated": False,
        "rollback_stack_mutated": False,
    })
    traces.append(trace)
    return HGMQDTWritePrep6Result(
        shared_slot_parity=shared,
        qh_sandbox=qh,
        rollback_binding_dry_run=rollback,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-6",
            "dry_run": True,
            "isolated_real_shared_slot_store_only": True,
            "qh_storage_record_sandbox_only": True,
            "rollback_snapshot_binding_preview_only": True,
            "live_write_executed": False,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
            "production_write_ready": False,
            "result_id": write_prep6_result_id("write_prep6", shared.harness_id, qh.sandbox_id, rollback.dry_run_id),
        },
    )
