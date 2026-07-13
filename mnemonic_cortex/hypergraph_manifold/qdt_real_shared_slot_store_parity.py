"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: qdt real shared slot store parity.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
Isolated real SharedSlotStore parity harness for WRITE-PREP-6.

The harness may instantiate a brand-new in-memory SharedSlotStore and call its
write_slot method for parity checking only. It never receives or mutates a live
QDT/WM store object.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep4_result import HGMQDTWritePrep4Result, RealContractObjectPreview
from .hgm_qdt_write_prep6_result import (
    HGMQDTWritePrep6Options,
    SharedSlotStoreParityHarnessResult,
    SharedSlotStoreParityRecord,
    write_prep6_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .validation import ValidationResult


def coerce_write_prep6_options(options: Optional[HGMQDTWritePrep6Options | Mapping[str, Any]] = None) -> HGMQDTWritePrep6Options:
    if options is None:
        return HGMQDTWritePrep6Options()
    if isinstance(options, HGMQDTWritePrep6Options):
        return options
    if isinstance(options, Mapping):
        allowed = {k: v for k, v in dict(options).items() if k in HGMQDTWritePrep6Options.__dataclass_fields__}
        return HGMQDTWritePrep6Options(**allowed)
    return HGMQDTWritePrep6Options()


def _previews_from_input(obj: Any) -> tuple[RealContractObjectPreview, ...]:
    if isinstance(obj, HGMQDTWritePrep4Result):
        return tuple(obj.contract_object_result.previews or tuple())
    if isinstance(obj, RealContractObjectPreview):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        return tuple(v for v in obj if isinstance(v, RealContractObjectPreview))
    return tuple()


def _content_vector_from_preview(preview: RealContractObjectPreview) -> tuple[float, ...]:
    object_trace = dict(preview.object_trace or {})
    content = object_trace.get("content_vector")
    if isinstance(content, (list, tuple)):
        return tuple(float(v) for v in content)
    # SystemWriteProposal.to_trace intentionally records only shape, not full
    # content. Rebuild a deterministic bounded vector for parity shape checks.
    dim = int(preview.content_shape[0]) if preview.content_shape else 1
    seed = int(write_prep_stable_hash(preview.proposal_id, preview.local_slot_id, length=8), 16)
    values = []
    for idx in range(dim):
        raw = ((seed >> (idx % 16)) & 0xFF) / 255.0
        values.append(float(raw if raw else (idx + 1) / (dim + 1)))
    return tuple(values)


def _torch_tensor(values: tuple[float, ...]):
    import torch
    return torch.tensor(list(values), dtype=torch.float32)


def _expected_fingerprint(values: tuple[float, ...]) -> str:
    try:
        from mnemonic_cortex.working_memory.wm_shared_slot_store import tensor_fingerprint
        return tensor_fingerprint(_torch_tensor(values))
    except Exception:
        return write_prep_stable_hash(*(f"{float(v):.6f}" for v in values), length=24)


def build_real_shared_slot_store_parity_harness(
    prep4_or_previews: Any,
    config=None,
    options: Optional[HGMQDTWritePrep6Options | Mapping[str, Any]] = None,
) -> SharedSlotStoreParityHarnessResult:
    """Build isolated SharedSlotStore parity records without live-store mutation."""
    opts = coerce_write_prep6_options(options)
    validation = ValidationResult()
    traces = []
    previews = _previews_from_input(prep4_or_previews)
    if not previews:
        validation.error("prep6.shared_slot_parity.empty_previews", "RealContractObjectPreview records are required", "prep4_or_previews")
    if len(previews) > opts.max_parity_records:
        validation.warning("prep6.shared_slot_parity.bounded_records", "preview count exceeded max_parity_records; records truncated", "previews")
    records: list[SharedSlotStoreParityRecord] = []
    for preview in sorted(previews[: opts.max_parity_records], key=lambda p: (p.local_slot_id, p.proposal_id)):
        blockers: list[str] = []
        isolated_constructed = False
        isolated_write_attempted = False
        observed_canonical = ""
        write_granted = False
        fingerprint = ""
        values = _content_vector_from_preview(preview)
        if len(values) <= 0 or len(values) > opts.max_tensor_dim:
            blockers.append("content vector length outside WRITE-PREP-6 bounds")
        if any(not math.isfinite(float(v)) for v in values):
            blockers.append("content vector contains NaN/Inf")
        if preview.write_permission and opts.require_write_permission_false:
            blockers.append("preview write_permission must remain False")
        if not opts.allow_isolated_real_shared_slot_store:
            blockers.append("isolated real SharedSlotStore parity disabled by options")
        if not blockers:
            try:
                from mnemonic_cortex.working_memory.wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig
                store = SharedSlotStore(SharedSlotStoreConfig(namespace="hgm_qdt_write_prep6", dim=len(values), require_write_permission=True))
                isolated_constructed = True
                isolated_write_attempted = True
                write_result = store.write_slot(
                    memory_type=preview.memory_type,
                    local_slot_id=preview.local_slot_id,
                    content=_torch_tensor(values),
                    owner=preview.memory_type,
                    geometry_map=preview.geometry_map,
                    depth_index=preview.depth_index,
                    confidence=preview.confidence,
                    write_permission=False,
                    metadata={"source": "WRITE-PREP-6 isolated parity harness", "proposal_id": preview.proposal_id},
                )
                observed_canonical = str(write_result.canonical_id)
                write_granted = bool(write_result.write_permission_granted)
                fingerprint = str(write_result.fingerprint)
            except Exception as exc:  # pragma: no cover - environment dependent
                blockers.append(str(exc))
                validation.error("prep6.shared_slot_parity.real_store_failed", f"isolated SharedSlotStore parity failed: {exc}", preview.proposal_id)
        if not fingerprint:
            fingerprint = _expected_fingerprint(values)
        parity_ok = bool(isolated_constructed and isolated_write_attempted and observed_canonical.startswith("css-") and not write_granted and not blockers)
        if not parity_ok:
            validation.warning("prep6.shared_slot_parity.not_ready", "SharedSlotStore parity record is not fully ready", preview.proposal_id)
        trace = trace_write_prep("qdt_real_shared_slot_store_parity.record", validation, {
            "proposal_id": preview.proposal_id,
            "isolated_store_constructed": isolated_constructed,
            "isolated_write_attempted": isolated_write_attempted,
            "parity_ok": parity_ok,
            "write_permission_granted": write_granted,
            "live_store_mutated": False,
            "blockers": tuple(blockers),
        })
        traces.append(trace)
        records.append(SharedSlotStoreParityRecord(
            parity_id=write_prep6_result_id("slot_parity", preview.preview_id, observed_canonical or preview.canonical_slot_id, parity_ok),
            source_preview_id=preview.preview_id,
            proposal_id=preview.proposal_id,
            wm_local_slot_id=preview.local_slot_id,
            expected_canonical_slot_id=preview.canonical_slot_id,
            observed_canonical_slot_id=observed_canonical,
            memory_type=preview.memory_type,
            geometry_map=preview.geometry_map,
            depth_index=preview.depth_index,
            vector_fingerprint=fingerprint,
            isolated_store_constructed=isolated_constructed,
            isolated_write_attempted=isolated_write_attempted,
            parity_ok=parity_ok,
            write_permission_granted=write_granted,
            blockers=tuple(blockers),
            trace_id=trace.trace_id,
            metadata={
                "stage": "HGM-QDT-WRITE-PREP-6",
                "isolated_shared_slot_store_only": True,
                "live_store_mutated": False,
                "system_commitgate_stage_called": False,
                "system_commitgate_commit_called": False,
            },
        ))
    final_trace = trace_write_prep("qdt_real_shared_slot_store_parity.build_real_shared_slot_store_parity_harness", validation, {
        "record_count": len(records),
        "parity_ok": bool(records and all(r.parity_ok for r in records)),
        "live_store_mutated": False,
    })
    traces.append(final_trace)
    return SharedSlotStoreParityHarnessResult(
        harness_id=write_prep6_result_id("shared_slot_parity_harness", tuple(r.parity_id for r in records)),
        parity_records=tuple(records),
        parity_ok=bool(records and all(r.parity_ok for r in records)),
        isolated_store_mutated=bool(records and any(r.isolated_write_attempted for r in records)),
        live_store_mutated=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-6",
            "isolated_real_shared_slot_store": True,
            "live_store_mutated": False,
        },
    )
