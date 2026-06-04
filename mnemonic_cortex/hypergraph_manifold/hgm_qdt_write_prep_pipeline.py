"""High-level HGM/QDT write-preparation pipeline.

This orchestrates read-only contract probes, tensor proposal previews, slot ID
mapping, q-spin/QH conversion previews, and rollback snapshot handshakes.
It never executes or stages QDT/WM writes.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm4_result import TraceSafeMemoryPlan
from .hgm_qdt_write_prep_result import HGMQDTWritePrepOptions, HGMQDTWritePrepResult, trace_write_prep, write_prep_stable_hash
from .qdt_proposal_materialization import build_proposal_materialization_contract
from .qdt_qspin_qh_contract import build_qspin_qh_conversion_contract
from .qdt_rollback_handshake import build_rollback_snapshot_handshake
from .qdt_slot_mapping_plan import build_slot_id_mapping_plan
from .qdt_write_contract_probe import _coerce_options, probe_qdt_wm_write_contracts
from .validation import ValidationResult


def build_hgm_qdt_write_prep_contracts(memory_plan: TraceSafeMemoryPlan | Any, config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> HGMQDTWritePrepResult:
    """Build all HGM->QDT/WM write-prep contracts in read-only mode."""
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    if not isinstance(memory_plan, TraceSafeMemoryPlan):
        validation.error("write_prep.invalid_memory_plan", "memory_plan must be a TraceSafeMemoryPlan", "memory_plan")
    probe = probe_qdt_wm_write_contracts(config=config, options=opts)
    validation.merge(probe.validation)
    traces.extend(probe.trace_records)
    proposal_contract = build_proposal_materialization_contract(memory_plan, config=config, options=opts)
    validation.merge(proposal_contract.validation)
    traces.extend(proposal_contract.trace_records)
    slot_plan = build_slot_id_mapping_plan(memory_plan, config=config, options=opts)
    validation.merge(slot_plan.validation)
    traces.extend(slot_plan.trace_records)
    qh_contract = build_qspin_qh_conversion_contract(memory_plan, config=config, options=opts)
    validation.merge(qh_contract.validation)
    traces.extend(qh_contract.trace_records)
    rollback = build_rollback_snapshot_handshake(memory_plan, config=config, options=opts)
    validation.merge(rollback.validation)
    traces.extend(rollback.trace_records)
    if not probe.compatible:
        validation.warning("write_prep.probe_not_fully_compatible", "QDT/WM signature probe is not fully compatible in this environment", "contract_probe")
    if not rollback.complete:
        validation.warning("write_prep.rollback_not_bound", "rollback handshake is not bound to actual WM snapshots yet", "rollback_handshake")
    trace = trace_write_prep("hgm_qdt_write_prep_pipeline.build_hgm_qdt_write_prep_contracts", validation, {
        "probe_compatible": probe.compatible,
        "tensor_previews": len(proposal_contract.tensor_previews),
        "slot_mappings": len(slot_plan.mappings),
        "qh_conversions": len(qh_contract.conversions),
        "rollback_requirements": len(rollback.requirements),
        "live_writes": False,
    })
    traces.append(trace)
    return HGMQDTWritePrepResult(
        contract_probe=probe,
        proposal_contract=proposal_contract,
        slot_mapping_plan=slot_plan,
        qspin_qh_contract=qh_contract,
        rollback_handshake=rollback,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-1",
            "read_only": True,
            "live_writes": False,
            "result_id": f"hgm_qdt_write_prep_1_{write_prep_stable_hash(getattr(memory_plan, 'plan_id', 'invalid'), len(traces))}",
        },
    )
