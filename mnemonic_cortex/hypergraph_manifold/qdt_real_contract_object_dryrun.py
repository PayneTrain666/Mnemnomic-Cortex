"""Real QDT/WM contract-object construction dry-run for WRITE-PREP-4.

This module may construct SystemWriteProposal objects locally for validation,
but it never stages or commits them.  The returned records contain trace-safe
previews rather than live mutable writer state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep2_result import DryRunSystemWriteProposalPreview, HGMQDTWritePrep2Result
from .hgm_qdt_write_prep3_result import HGMQDTWritePrep3Result
from .hgm_qdt_write_prep4_result import (
    HGMQDTWritePrep4Options,
    RealContractObjectConstructionResult,
    RealContractObjectPreview,
    write_prep4_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash, write_prep_redact
from .validation import ValidationResult


def coerce_write_prep4_options(options: Optional[HGMQDTWritePrep4Options | Mapping[str, Any]] = None) -> HGMQDTWritePrep4Options:
    if options is None:
        return HGMQDTWritePrep4Options()
    if isinstance(options, HGMQDTWritePrep4Options):
        return options
    if isinstance(options, Mapping):
        allowed = {k: v for k, v in dict(options).items() if k in HGMQDTWritePrep4Options.__dataclass_fields__}
        return HGMQDTWritePrep4Options(**allowed)
    return HGMQDTWritePrep4Options()


def _proposals_from_input(obj: Any) -> tuple[DryRunSystemWriteProposalPreview, ...]:
    if isinstance(obj, HGMQDTWritePrep2Result):
        return tuple(obj.proposal_builder_result.proposals or tuple())
    if isinstance(obj, HGMQDTWritePrep3Result):
        return tuple(getattr(obj.commitgate_simulation.sandbox_after, "metadata", {}).get("_proposals", tuple()) or tuple())
    if isinstance(obj, DryRunSystemWriteProposalPreview):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, DryRunSystemWriteProposalPreview))
    # HGM-QDT-WRITE-PREP-3 normally carries operations, not full proposal objects;
    # callers should pass WRITE-PREP-2 or proposal previews for construction.
    return tuple()


def _torch_tensor(values: tuple[float, ...]):
    import torch
    return torch.tensor(list(values), dtype=torch.float32)


def _construct_system_write_proposal(proposal: DryRunSystemWriteProposalPreview):
    from mnemonic_cortex.working_memory.wm_system_commit_gate import SystemWriteProposal
    tensor = _torch_tensor(tuple(float(v) for v in proposal.content_vector))
    obj = SystemWriteProposal(
        proposal_id=f"sysprop-{write_prep_stable_hash(proposal.proposal_id, proposal.local_slot_id, length=24)}",
        content=tensor,
        memory_type=proposal.memory_type,
        local_slot_id=proposal.local_slot_id,
        geometry_map=proposal.geometry_map,
        depth_index=proposal.depth_index,
        triplet_index=proposal.triplet_index,
        bank_name=proposal.bank_name,
        task_mode=proposal.task_mode,
        confidence=proposal.confidence,
        write_permission=False,
        metadata={"source": "HGM-QDT-WRITE-PREP-4", "dry_run_contract_object": True, "source_dry_run_proposal_id": proposal.proposal_id},
    )
    obj.validate(len(proposal.content_vector))
    return obj


def build_real_contract_object_previews(
    prep2_result_or_proposals: Any,
    config=None,
    options: Optional[HGMQDTWritePrep4Options | Mapping[str, Any]] = None,
) -> RealContractObjectConstructionResult:
    """Construct SystemWriteProposal-shaped objects in dry-run preview mode.

    No returned preview has write_permission=True.  No SystemCommitGate object is
    called here.  If Torch or WM contract imports are unavailable, the function
    fails closed with explicit blockers while still returning typed results.
    """
    opts = coerce_write_prep4_options(options)
    validation = ValidationResult()
    traces = []
    proposals = _proposals_from_input(prep2_result_or_proposals)
    if not proposals:
        validation.warning("prep4.contract.empty_proposals", "no dry-run proposal previews supplied", "prep2_result_or_proposals")
    if len(proposals) > opts.max_contract_objects:
        validation.warning("prep4.contract.bounded_object_count", "proposal count exceeded max_contract_objects; previews truncated", "proposals")
    previews: list[RealContractObjectPreview] = []
    for proposal in tuple(sorted(proposals[: opts.max_contract_objects], key=lambda p: (p.local_slot_id, p.proposal_id))):
        blockers: list[str] = []
        constructed = False
        object_trace: Mapping[str, Any] = {}
        class_name = "SystemWriteProposal"
        validation_ready = False
        if len(proposal.content_vector) <= 0 or len(proposal.content_vector) > opts.max_tensor_dim:
            blockers.append("content vector length outside WRITE-PREP-4 bounds")
        if proposal.write_permission:
            blockers.append("dry-run proposal unexpectedly has live write_permission=True")
        if not opts.allow_real_contract_construction:
            blockers.append("real contract construction disabled by options")
        if not blockers:
            try:
                obj = _construct_system_write_proposal(proposal)
                object_trace = write_prep_redact("object_trace", obj.to_trace())
                constructed = True
                validation_ready = True
            except Exception as exc:  # pragma: no cover - environment dependent
                if opts.require_torch_for_real_contracts:
                    validation.error("prep4.contract.construction_failed", f"SystemWriteProposal dry-run construction failed: {exc}", proposal.proposal_id)
                else:
                    validation.warning("prep4.contract.construction_unavailable", f"SystemWriteProposal dry-run construction unavailable: {exc}", proposal.proposal_id)
                blockers.append(str(exc))
        trace = trace_write_prep("qdt_real_contract_object_dryrun.real_contract_preview", validation, {
            "source_proposal_id": proposal.proposal_id,
            "constructed": constructed,
            "write_permission": False,
            "blockers": tuple(blockers),
            "object_trace": object_trace,
        })
        traces.append(trace)
        previews.append(RealContractObjectPreview(
            preview_id=write_prep4_result_id("real_contract_preview", proposal.proposal_id, constructed),
            source_proposal_id=proposal.proposal_id,
            constructed=constructed,
            contract_class_name=class_name,
            proposal_id=str(object_trace.get("proposal_id", f"sysprop-{write_prep_stable_hash(proposal.proposal_id, proposal.local_slot_id, length=24)}")),
            content_shape=tuple(object_trace.get("content_shape", proposal.content_shape)),
            memory_type=proposal.memory_type,
            local_slot_id=proposal.local_slot_id,
            canonical_slot_id=proposal.canonical_slot_id,
            geometry_map=proposal.geometry_map,
            depth_index=proposal.depth_index,
            triplet_index=proposal.triplet_index,
            bank_name=proposal.bank_name,
            task_mode=proposal.task_mode,
            confidence=proposal.confidence,
            write_permission=False,
            validation_ready=validation_ready,
            object_trace=object_trace,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-4",
                "dry_run": True,
                "real_contract_object_constructed": constructed,
                "system_commitgate_stage_called": False,
                "system_commitgate_commit_called": False,
                "live_write_executed": False,
            },
        ))
    final_trace = trace_write_prep("qdt_real_contract_object_dryrun.build_real_contract_object_previews", validation, {
        "preview_count": len(previews),
        "constructed_count": sum(1 for p in previews if p.constructed),
        "live_write_executed": False,
    })
    traces.append(final_trace)
    return RealContractObjectConstructionResult(
        previews=tuple(previews),
        constructed_count=sum(1 for p in previews if p.constructed),
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-4",
            "dry_run": True,
            "contract_object_preview_only": True,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
        },
    )
