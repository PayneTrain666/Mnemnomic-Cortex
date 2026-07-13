"""
Plain-language summary
----------------------
What this file is for: Dry-run or sandbox helper (qdt_dry_run_proposal_builder).
How it fits in the system: Lets engineers rehearse a path safely without committing live side effects.
Status: LOW-USE / SAFETY SCAFFOLD
Important notes for non-coders: Not the everyday training path.

Technical notes (original):
Dry-run SystemWriteProposal preview builder for HGM/QDT integration.

The builder creates proposal-shaped preview records from WRITE-PREP-1 contracts.
It does not instantiate/stage/commit live SystemWriteProposal objects. Optional
Torch validation is used only to verify tensor shape/finite values when Torch is
available and allowed.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepResult,
    ProposalMaterializationContract,
    QSpinQHConversionContract,
    SlotIDMappingPlan,
    TensorProposalPreview,
    trace_write_prep,
    write_prep_redact,
    write_prep_stable_hash,
)
from .hgm_qdt_write_prep2_result import (
    DryRunProposalBuilderResult,
    DryRunSystemWriteProposalPreview,
    HGMQDTWritePrep2Options,
)
from .qdt_write_contract_probe import _coerce_options as _coerce_prep1_options
from .validation import ValidationResult


def _coerce_options(options: Optional[HGMQDTWritePrep2Options | Mapping[str, Any]]) -> HGMQDTWritePrep2Options:
    if options is None:
        return HGMQDTWritePrep2Options()
    if isinstance(options, HGMQDTWritePrep2Options):
        return options
    return HGMQDTWritePrep2Options(**dict(options))


def _torch_validation(vector: tuple[float, ...], shape: tuple[int, ...], allow: bool) -> tuple[bool, str]:
    if not allow:
        return False, "torch validation disabled by options"
    try:
        import torch  # type: ignore
        tensor = torch.tensor(vector, dtype=torch.float32)
        if tuple(tensor.shape) != tuple(shape):
            return False, "torch tensor shape mismatch"
        if not bool(torch.isfinite(tensor).all().item()):
            return False, "torch tensor contains NaN/Inf"
        return True, "torch validation passed"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"torch unavailable or validation skipped: {exc}"


def _finite_vector(values: tuple[float, ...]) -> bool:
    return bool(values) and all(math.isfinite(float(v)) for v in values)


def _mapping_by_payload(slot_plan: SlotIDMappingPlan) -> dict[str, Any]:
    out = {}
    for mapping in slot_plan.mappings:
        out[str(mapping.source_hook_id)] = mapping
        out[str(mapping.hgm_target_slot_id)] = mapping
        out[str(mapping.metadata.get("source_record_id", ""))] = mapping
    return out


def _conversion_by_qspin(qh_contract: QSpinQHConversionContract) -> dict[str, Any]:
    out = {}
    for conv in qh_contract.conversions:
        out[str(conv.qspin_signature_id)] = conv
        out[str(conv.metadata.get("source_record_id", ""))] = conv
    return out


def _contracts_from_input(write_prep_result_or_contracts: Any) -> tuple[ProposalMaterializationContract | None, SlotIDMappingPlan | None, QSpinQHConversionContract | None, Any | None]:
    if isinstance(write_prep_result_or_contracts, HGMQDTWritePrepResult):
        return (
            write_prep_result_or_contracts.proposal_contract,
            write_prep_result_or_contracts.slot_mapping_plan,
            write_prep_result_or_contracts.qspin_qh_contract,
            write_prep_result_or_contracts.rollback_handshake,
        )
    proposal_contract = getattr(write_prep_result_or_contracts, "proposal_contract", None)
    slot_plan = getattr(write_prep_result_or_contracts, "slot_mapping_plan", None)
    qh_contract = getattr(write_prep_result_or_contracts, "qspin_qh_contract", None)
    rollback = getattr(write_prep_result_or_contracts, "rollback_handshake", None)
    return (proposal_contract, slot_plan, qh_contract, rollback)


def build_dry_run_system_write_proposal_previews(write_prep_result_or_contracts: Any, config=None, options: Optional[HGMQDTWritePrep2Options | Mapping[str, Any]] = None) -> DryRunProposalBuilderResult:
    """Build proposal-shaped preview records without constructing live writes."""
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    proposal_contract, slot_plan, qh_contract, rollback = _contracts_from_input(write_prep_result_or_contracts)
    if proposal_contract is None or not isinstance(proposal_contract, ProposalMaterializationContract):
        validation.error("prep2.missing_proposal_contract", "ProposalMaterializationContract is required", "proposal_contract")
        trace = trace_write_prep("qdt_dry_run_proposal_builder.missing_contract", validation, {"live_writes": False})
        return DryRunProposalBuilderResult(tuple(), validation, (trace,), {"live_writes": False})
    if slot_plan is None or not isinstance(slot_plan, SlotIDMappingPlan):
        validation.error("prep2.missing_slot_mapping", "SlotIDMappingPlan is required", "slot_mapping_plan")
    if qh_contract is None or not isinstance(qh_contract, QSpinQHConversionContract):
        validation.error("prep2.missing_qh_contract", "QSpinQHConversionContract is required", "qspin_qh_contract")

    tensor_previews = tuple(proposal_contract.tensor_previews or tuple())
    if not tensor_previews:
        validation.warning("prep2.empty_tensor_previews", "no tensor previews supplied", "tensor_previews")
    if len(tensor_previews) > opts.max_proposals:
        validation.warning("prep2.bounded_proposals", "proposal preview count exceeded max_proposals; truncated", "tensor_previews")
    mapping_by_payload = _mapping_by_payload(slot_plan) if isinstance(slot_plan, SlotIDMappingPlan) else {}
    qh_by_qspin = _conversion_by_qspin(qh_contract) if isinstance(qh_contract, QSpinQHConversionContract) else {}
    proposals = []
    for preview in sorted(tensor_previews[: opts.max_proposals], key=lambda p: (p.source_payload_id, p.preview_id)):
        blockers = []
        vector = tuple(float(v) for v in preview.content_vector)
        shape = tuple(int(v) for v in preview.content_shape)
        if len(vector) > opts.max_tensor_dim:
            blockers.append("tensor dimension exceeds max_tensor_dim")
            validation.error("prep2.tensor_dim_exceeded", "tensor dimension exceeds max_tensor_dim", preview.preview_id)
        if not _finite_vector(vector):
            blockers.append("content vector is empty or non-finite")
            validation.error("prep2.invalid_tensor_preview", "content vector is empty or non-finite", preview.preview_id)
        if shape != (len(vector),):
            blockers.append("content shape does not match vector length")
            validation.error("prep2.shape_mismatch", "content shape does not match vector length", preview.preview_id)
        mapping = mapping_by_payload.get(preview.source_payload_id)
        if mapping is None and isinstance(slot_plan, SlotIDMappingPlan) and len(slot_plan.mappings) == 1:
            # WRITE-PREP-1 historically keyed mapping records by hook/target IDs,
            # while tensor previews key by payload ID.  A single mapping is an
            # unambiguous read-only preview fallback; multiple mappings still
            # fail closed unless an exact key exists.
            mapping = slot_plan.mappings[0]
            validation.warning("prep2.slot_mapping_singleton_fallback", "using sole slot mapping as dry-run preview fallback", preview.source_payload_id)
        if mapping is None:
            blockers.append("missing slot mapping")
            validation.error("prep2.missing_slot_mapping_for_payload", "missing slot mapping for tensor preview", preview.source_payload_id)
            local_slot_id = f"wm_missing_{write_prep_stable_hash(preview.source_payload_id)}"
            canonical_slot_id = ""
        else:
            local_slot_id = mapping.wm_local_slot_id
            canonical_slot_id = mapping.wm_canonical_slot_id
        # Find QH conversion. If exact source lookup fails, use first conversion as conservative preview.
        qh_conv = None
        if isinstance(qh_contract, QSpinQHConversionContract):
            for conv in qh_contract.conversions:
                if conv.metadata.get("source_record_id") == preview.source_payload_id:
                    qh_conv = conv
                    break
            if qh_conv is None and qh_contract.conversions:
                qh_conv = qh_contract.conversions[0]
        if qh_conv is None:
            blockers.append("missing QH conversion")
            validation.error("prep2.missing_qh_conversion", "missing q-spin/QH conversion for proposal", preview.source_payload_id)
            memory_type = "wm"
            geometry_map = "unknown"
            depth_index = 0
            triplet_index = 0
            bank_name = "qdt_working_memory"
            task_mode = "quantum_holographic"
            qspin_id = ""
        else:
            memory_type = qh_conv.memory_type
            geometry_map = qh_conv.geometry_map
            depth_index = qh_conv.depth_index
            triplet_index = qh_conv.triplet_index
            bank_name = qh_conv.bank_name
            task_mode = qh_conv.task_mode
            qspin_id = qh_conv.qspin_signature_id
        torch_ok, torch_reason = _torch_validation(vector, shape, opts.allow_torch_validation)
        if not torch_ok:
            validation.warning("prep2.torch_validation_unavailable", torch_reason, preview.preview_id)
        # Proposal preview is ready if the contract data is internally complete, even if Torch is unavailable.
        qdt_validation_ready = bool(torch_ok or _finite_vector(vector))
        confidence = 1.0 if not blockers else 0.0
        proposal_id = f"drysysprop-{write_prep_stable_hash(preview.preview_id, preview.source_payload_id, local_slot_id)}"
        trace = trace_write_prep("qdt_dry_run_proposal_builder.proposal_preview", validation, {
            "proposal_id": proposal_id,
            "source_preview_id": preview.preview_id,
            "write_permission": False,
            "simulated_write_permission": opts.simulated_write_permission,
            "blockers": blockers,
            "metadata": write_prep_redact("metadata", dict(preview.metadata or {})),
        })
        traces.append(trace)
        proposals.append(DryRunSystemWriteProposalPreview(
            proposal_id=proposal_id,
            source_preview_id=preview.preview_id,
            source_payload_id=preview.source_payload_id,
            content_shape=shape,
            content_vector=vector,
            content_fingerprint=preview.content_fingerprint,
            memory_type=memory_type,
            local_slot_id=local_slot_id,
            canonical_slot_id=canonical_slot_id,
            geometry_map=geometry_map,
            depth_index=depth_index,
            triplet_index=triplet_index,
            bank_name=bank_name,
            task_mode=task_mode,
            confidence=confidence,
            write_permission=False,
            simulated_write_permission=bool(opts.simulated_write_permission),
            tensor_available=torch_ok,
            qdt_validation_ready=qdt_validation_ready,
            ready=bool(not blockers and qdt_validation_ready),
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "dry_run": True,
                "preview_only": True,
                "stage_called": False,
                "commit_called": False,
                "torch_validation_reason": torch_reason,
                "qspin_signature_id": qspin_id,
            },
        ))
    final_trace = trace_write_prep("qdt_dry_run_proposal_builder.build_dry_run_system_write_proposal_previews", validation, {
        "proposal_count": len(proposals),
        "live_writes": False,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return DryRunProposalBuilderResult(
        proposals=tuple(proposals),
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-2",
            "dry_run": True,
            "live_writes": False,
            "stage_called": False,
            "commit_called": False,
            "proposal_count": len(proposals),
        },
    )
