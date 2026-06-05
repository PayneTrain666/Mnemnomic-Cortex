"""QHStorageRecord sandbox construction for WRITE-PREP-6.

This module constructs QHStorageRecord-compatible objects in an isolated local
sandbox for schema/shape validation. It never mutates live QuantumHolographicStorage.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result, RealContractObjectPreview
from .hgm_qdt_write_prep6_result import (
    HGMQDTWritePrep6Options,
    QHStorageRecordSandboxRecord,
    QHStorageRecordSandboxResult,
    SharedSlotStoreParityHarnessResult,
    write_prep6_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .qdt_real_shared_slot_store_parity import coerce_write_prep6_options, _content_vector_from_preview, _expected_fingerprint
from .validation import ValidationResult


def _previews_from_input(obj: Any) -> tuple[RealContractObjectPreview, ...]:
    if isinstance(obj, HGMQDTWritePrep4Result):
        return tuple(obj.contract_object_result.previews or tuple())
    if isinstance(obj, RealContractObjectPreview):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, RealContractObjectPreview))
    return tuple()


def _parity_by_preview(parity: Any) -> dict[str, Any]:
    if isinstance(parity, SharedSlotStoreParityHarnessResult):
        return {rec.source_preview_id: rec for rec in parity.parity_records}
    return {}


def _torch_tensor(values: tuple[float, ...]):
    import torch
    return torch.tensor(list(values), dtype=torch.float32)


def build_qh_storage_record_sandbox(
    prep4_or_previews: Any,
    shared_slot_parity: Any = None,
    config=None,
    options: Optional[HGMQDTWritePrep6Options | Mapping[str, Any]] = None,
) -> QHStorageRecordSandboxResult:
    """Construct isolated QHStorageRecord-compatible records for validation."""
    opts = coerce_write_prep6_options(options)
    validation = ValidationResult()
    traces = []
    previews = _previews_from_input(prep4_or_previews)
    parity_map = _parity_by_preview(shared_slot_parity)
    if not previews:
        validation.error("prep6.qh_sandbox.empty_previews", "RealContractObjectPreview records are required", "prep4_or_previews")
    if len(previews) > opts.max_qh_records:
        validation.warning("prep6.qh_sandbox.bounded_records", "preview count exceeded max_qh_records; records truncated", "previews")
    records: list[QHStorageRecordSandboxRecord] = []
    for preview in sorted(previews[: opts.max_qh_records], key=lambda p: (p.local_slot_id, p.proposal_id)):
        blockers: list[str] = []
        constructed = False
        validated = False
        values = _content_vector_from_preview(preview)
        if len(values) <= 0 or len(values) > opts.max_tensor_dim:
            blockers.append("content vector length outside WRITE-PREP-6 bounds")
        if any(not math.isfinite(float(v)) for v in values):
            blockers.append("content vector contains NaN/Inf")
        if preview.write_permission and opts.require_write_permission_false:
            blockers.append("preview write_permission must remain False")
        if not opts.allow_qh_storage_record_sandbox:
            blockers.append("QHStorageRecord sandbox construction disabled by options")
        parity = parity_map.get(preview.preview_id)
        canonical_slot_id = str(getattr(parity, "observed_canonical_slot_id", "") or preview.canonical_slot_id)
        if not canonical_slot_id.startswith("css-"):
            blockers.append("canonical slot id must start with css-")
        composite_code = ""
        qh_record_id = f"qhrec-{write_prep_stable_hash(preview.proposal_id, canonical_slot_id, length=24)}"
        fingerprint = _expected_fingerprint(values)
        vector_norm = math.sqrt(sum(float(v) * float(v) for v in values)) if values else 0.0
        if not blockers:
            try:
                from mnemonic_cortex.working_memory.wm_quantum_holographic_storage import QHStorageRecord, build_qh_code_schema
                schema = build_qh_code_schema(
                    depth_index=preview.depth_index,
                    bank_name=preview.bank_name,
                    geometry_name=preview.geometry_map,
                    triplet_index=preview.triplet_index,
                    memory_type=preview.memory_type,
                    task_mode=preview.task_mode,
                )
                composite_code = schema.composite_code()
                record = QHStorageRecord(
                    record_id=qh_record_id,
                    canonical_slot_id=canonical_slot_id,
                    code_schema=schema,
                    vector_fingerprint=fingerprint,
                    vector_norm=float(vector_norm),
                    confidence=preview.confidence,
                    write_permission_required=True,
                    write_permission_granted=False,
                    interference=None,
                    metadata={"source": "WRITE-PREP-6 QHStorageRecord sandbox", "proposal_id": preview.proposal_id},
                )
                record.validate()
                constructed = True
                validated = True
            except Exception as exc:  # pragma: no cover - environment dependent
                blockers.append(str(exc))
                validation.error("prep6.qh_sandbox.construction_failed", f"QHStorageRecord sandbox construction failed: {exc}", preview.proposal_id)
        if not composite_code:
            composite_code = f"qh-{write_prep_stable_hash(preview.depth_index, preview.bank_name, preview.geometry_map, preview.triplet_index, preview.memory_type, preview.task_mode)}"
        if not validated:
            validation.warning("prep6.qh_sandbox.not_validated", "QHStorageRecord sandbox record is not validated", preview.proposal_id)
        trace = trace_write_prep("qdt_qh_storage_record_sandbox.record", validation, {
            "proposal_id": preview.proposal_id,
            "constructed": constructed,
            "validated": validated,
            "qh_record_id": qh_record_id,
            "live_qh_storage_mutated": False,
            "blockers": tuple(blockers),
        })
        traces.append(trace)
        records.append(QHStorageRecordSandboxRecord(
            sandbox_record_id=write_prep6_result_id("qh_sandbox_record", preview.preview_id, qh_record_id, validated),
            source_preview_id=preview.preview_id,
            proposal_id=preview.proposal_id,
            canonical_slot_id=canonical_slot_id,
            qh_record_id=qh_record_id,
            composite_code=composite_code,
            vector_fingerprint=fingerprint,
            vector_norm=vector_norm,
            depth_index=preview.depth_index,
            geometry_map=preview.geometry_map,
            triplet_index=preview.triplet_index,
            memory_type=preview.memory_type,
            task_mode=preview.task_mode,
            confidence=preview.confidence,
            write_permission_granted=False,
            constructed=constructed,
            validated=validated,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-6",
                "sandbox_record_only": True,
                "qh_storage_mutated": False,
                "live_qh_storage_mutated": False,
                "interference_check_live": False,
            },
        ))
    final_trace = trace_write_prep("qdt_qh_storage_record_sandbox.build_qh_storage_record_sandbox", validation, {
        "record_count": len(records),
        "validated_count": sum(1 for r in records if r.validated),
        "live_qh_storage_mutated": False,
    })
    traces.append(final_trace)
    return QHStorageRecordSandboxResult(
        sandbox_id=write_prep6_result_id("qh_storage_sandbox", tuple(r.sandbox_record_id for r in records)),
        records=tuple(records),
        constructed_count=sum(1 for r in records if r.constructed),
        validated_count=sum(1 for r in records if r.validated),
        sandbox_qh_storage_mutated=bool(records and any(r.constructed for r in records)),
        live_qh_storage_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-6",
            "sandbox_qh_storage_records_only": True,
            "live_qh_storage_mutated": False,
        },
    )
