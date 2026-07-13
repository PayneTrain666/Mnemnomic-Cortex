"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt end to end write simulation.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Non-mutating end-to-end HGM/QDT write simulation.

The simulation uses WRITE-PREP-1 contracts to produce dry-run proposal previews,
CommitGate-style preflight checks, and readiness reports. It does not stage,
commit, write shared slots, create QH records, or mutate rollback stacks.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep_result import HGMQDTWritePrepResult, trace_write_prep, write_prep_stable_hash
from .hgm_qdt_write_prep2_result import EndToEndWriteSimulationReport, HGMQDTWritePrep2Options, HGMQDTWritePrep2Result
from .qdt_commitgate_preflight import run_commitgate_preflight
from .qdt_dry_run_proposal_builder import _coerce_options, build_dry_run_system_write_proposal_previews
from .validation import ValidationResult


def simulate_end_to_end_hgm_qdt_write(write_prep_result: Any, config=None, options: Optional[HGMQDTWritePrep2Options | Mapping[str, Any]] = None) -> EndToEndWriteSimulationReport:
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(write_prep_result, HGMQDTWritePrepResult):
        validation.error("prep2.invalid_write_prep_result", "HGMQDTWritePrepResult is required", "write_prep_result")
    builder = build_dry_run_system_write_proposal_previews(write_prep_result, config=config, options=opts)
    validation.merge(builder.validation)
    traces.extend(builder.trace_records)
    preflight = run_commitgate_preflight(builder, write_prep_result=write_prep_result if isinstance(write_prep_result, HGMQDTWritePrepResult) else None, config=config, options=opts)
    validation.merge(preflight.validation)
    traces.extend(preflight.trace_records)
    if isinstance(write_prep_result, HGMQDTWritePrepResult):
        slot_ready = bool(write_prep_result.slot_mapping_plan.mappings)
        qh_ready = bool(write_prep_result.qspin_qh_contract.conversions)
        rollback_ready = bool(write_prep_result.rollback_handshake.complete)
    else:
        slot_ready = False
        qh_ready = False
        rollback_ready = False
    # Rollback readiness is intentionally a hard limiter for write simulation readiness.
    readiness_score = preflight.readiness_score
    if not rollback_ready:
        readiness_score = min(readiness_score, opts.conservative_missing_score)
        validation.warning("prep2.rollback_not_ready", "rollback handshake is not bound to real snapshots; write simulation not commit-ready", "rollback_handshake")
    report_id = f"hgm_qdt_write_sim_{write_prep_stable_hash(getattr(write_prep_result, 'metadata', {}).get('result_id', 'unknown'), len(builder.proposals), preflight.readiness_score)}"
    trace = trace_write_prep("qdt_end_to_end_write_simulation.simulate_end_to_end_hgm_qdt_write", validation, {
        "report_id": report_id,
        "slot_ready": slot_ready,
        "qh_ready": qh_ready,
        "rollback_ready": rollback_ready,
        "readiness_score": readiness_score,
        "live_write_executed": False,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(trace)
    return EndToEndWriteSimulationReport(
        report_id=report_id,
        proposal_builder_result=builder,
        preflight_result=preflight,
        slot_ready=slot_ready,
        qh_ready=qh_ready,
        rollback_ready=rollback_ready,
        readiness_score=readiness_score,
        live_write_executed=False,
        stage_called=False,
        commit_called=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-2",
            "dry_run": True,
            "non_mutating": True,
            "no_system_commit_gate_stage": True,
            "no_system_commit_gate_commit": True,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
        },
    )


def build_hgm_qdt_write_prep_2(write_prep_result: Any, config=None, options: Optional[HGMQDTWritePrep2Options | Mapping[str, Any]] = None) -> HGMQDTWritePrep2Result:
    validation = ValidationResult()
    traces = []
    report = simulate_end_to_end_hgm_qdt_write(write_prep_result, config=config, options=options)
    validation.merge(report.validation)
    traces.extend(report.trace_records)
    trace = trace_write_prep("qdt_end_to_end_write_simulation.build_hgm_qdt_write_prep_2", validation, {
        "report_id": report.report_id,
        "proposal_count": len(report.proposal_builder_result.proposals),
        "preflight_ready": report.preflight_result.ready,
        "live_write_executed": False,
    })
    traces.append(trace)
    return HGMQDTWritePrep2Result(
        proposal_builder_result=report.proposal_builder_result,
        preflight_result=report.preflight_result,
        simulation_report=report,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-2",
            "dry_run": True,
            "read_only": True,
            "live_write_executed": False,
            "stage_called": False,
            "commit_called": False,
            "result_id": f"hgm_qdt_write_prep_2_{write_prep_stable_hash(report.report_id)}",
        },
    )
