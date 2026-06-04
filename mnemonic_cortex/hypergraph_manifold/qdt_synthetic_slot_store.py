"""Synthetic SharedSlotStore sandbox for HGM/QDT WRITE-PREP-3.

The sandbox is an immutable record of slot-like states used only for simulation.
It never calls or mutates the real SharedSlotStore.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

from .hgm_qdt_write_prep2_result import DryRunSystemWriteProposalPreview
from .hgm_qdt_write_prep3_result import (
    HGMQDTWritePrep3Options,
    SyntheticSharedSlotStoreSandbox,
    SyntheticSlotRecord,
    write_prep3_result_id,
)
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .qdt_dry_run_proposal_builder import _coerce_options as _coerce_prep2_options
from .validation import ValidationResult


def coerce_write_prep3_options(options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]]) -> HGMQDTWritePrep3Options:
    if options is None:
        return HGMQDTWritePrep3Options()
    if isinstance(options, HGMQDTWritePrep3Options):
        return options
    return HGMQDTWritePrep3Options(**dict(options))


def _fingerprint_vector(values: Sequence[float]) -> str:
    return write_prep_stable_hash(tuple(round(float(v), 12) for v in values), length=24)


def _zero_vector(length: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(max(0, int(length))))


def _proposal_slots(proposals: Iterable[DryRunSystemWriteProposalPreview], validation: ValidationResult, max_payload_values: int) -> tuple[SyntheticSlotRecord, ...]:
    slots: list[SyntheticSlotRecord] = []
    seen: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, DryRunSystemWriteProposalPreview):
            validation.warning("prep3.synthetic_store.skip_non_proposal", "non-proposal record skipped", "proposals")
            continue
        slot_id = str(proposal.local_slot_id or proposal.canonical_slot_id or proposal.proposal_id)
        if slot_id in seen:
            continue
        seen.add(slot_id)
        payload_len = len(proposal.content_vector)
        if payload_len > max_payload_values:
            validation.error("prep3.synthetic_store.payload_too_large", "proposal payload exceeds max_slot_payload_values", slot_id)
            payload_len = max_payload_values
        previous = _zero_vector(payload_len)
        previous_fp = _fingerprint_vector(previous)
        trace = trace_write_prep("qdt_synthetic_slot_store.initial_slot", validation, {
            "slot_id": slot_id,
            "canonical_slot_id": proposal.canonical_slot_id,
            "live_store_mutated": False,
        })
        slots.append(SyntheticSlotRecord(
            slot_id=slot_id,
            canonical_slot_id=str(proposal.canonical_slot_id or slot_id),
            current_vector=previous,
            current_fingerprint=previous_fp,
            previous_vector=previous,
            previous_fingerprint=previous_fp,
            version=0,
            trace_id=trace.trace_id,
            metadata={"source_proposal_id": proposal.proposal_id, "synthetic_only": True, "live_store_mutated": False},
        ))
    return tuple(sorted(slots, key=lambda s: (s.slot_id, s.canonical_slot_id)))


def build_synthetic_shared_slot_store_sandbox(
    proposals: Iterable[DryRunSystemWriteProposalPreview],
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> SyntheticSharedSlotStoreSandbox:
    """Build an immutable synthetic slot-store baseline from proposal previews."""
    opts = coerce_write_prep3_options(options)
    validation = ValidationResult()
    slots = _proposal_slots(tuple(proposals or tuple()), validation, opts.max_slot_payload_values)
    if not slots:
        validation.warning("prep3.synthetic_store.empty", "no synthetic slots were created", "proposals")
    trace = trace_write_prep("qdt_synthetic_slot_store.build_synthetic_shared_slot_store_sandbox", validation, {
        "slot_count": len(slots),
        "synthetic_only": True,
        "live_store_mutated": False,
    })
    return SyntheticSharedSlotStoreSandbox(
        sandbox_id=write_prep3_result_id("synthetic_slot_store", tuple(slot.slot_id for slot in slots)),
        slots=slots,
        validation=validation,
        trace_records=(trace,),
        metadata={"synthetic_only": True, "live_store_mutated": False, "slot_count": len(slots)},
    )


def apply_synthetic_slot_writes(
    sandbox: SyntheticSharedSlotStoreSandbox,
    proposals: Iterable[DryRunSystemWriteProposalPreview],
    committed_proposal_ids: Iterable[str],
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> SyntheticSharedSlotStoreSandbox:
    """Return a new synthetic sandbox after simulated proposal writes."""
    opts = coerce_write_prep3_options(options)
    validation = ValidationResult()
    validation.merge(sandbox.validation)
    proposal_by_slot = {str(p.local_slot_id or p.canonical_slot_id): p for p in tuple(proposals or tuple()) if isinstance(p, DryRunSystemWriteProposalPreview)}
    committed = set(str(v) for v in committed_proposal_ids or tuple())
    new_slots: list[SyntheticSlotRecord] = []
    traces = list(sandbox.trace_records)
    for slot in sandbox.slots:
        proposal = proposal_by_slot.get(slot.slot_id)
        if proposal is None or proposal.proposal_id not in committed:
            new_slots.append(slot)
            continue
        vector = tuple(float(v) for v in proposal.content_vector[: opts.max_slot_payload_values])
        if not vector or not all(math.isfinite(v) for v in vector):
            validation.error("prep3.synthetic_store.invalid_write_vector", "synthetic write vector must be finite and non-empty", proposal.proposal_id)
            new_slots.append(slot)
            continue
        trace = trace_write_prep("qdt_synthetic_slot_store.apply_synthetic_slot_write", validation, {
            "proposal_id": proposal.proposal_id,
            "slot_id": slot.slot_id,
            "synthetic_write": True,
            "live_store_mutated": False,
        })
        traces.append(trace)
        new_slots.append(SyntheticSlotRecord(
            slot_id=slot.slot_id,
            canonical_slot_id=slot.canonical_slot_id,
            current_vector=vector,
            current_fingerprint=_fingerprint_vector(vector),
            previous_vector=slot.current_vector,
            previous_fingerprint=slot.current_fingerprint,
            version=slot.version + 1,
            trace_id=trace.trace_id,
            metadata={**dict(slot.metadata or {}), "last_synthetic_proposal_id": proposal.proposal_id, "synthetic_mutated": True, "live_store_mutated": False},
        ))
    final_trace = trace_write_prep("qdt_synthetic_slot_store.apply_synthetic_slot_writes", validation, {
        "slot_count": len(new_slots),
        "committed_count": len(committed),
        "live_store_mutated": False,
    })
    traces.append(final_trace)
    return SyntheticSharedSlotStoreSandbox(
        sandbox_id=write_prep3_result_id("synthetic_slot_store_after", sandbox.sandbox_id, tuple(sorted(committed))),
        slots=tuple(sorted(new_slots, key=lambda s: (s.slot_id, s.canonical_slot_id))),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"synthetic_only": True, "synthetic_mutated": bool(committed), "live_store_mutated": False, "slot_count": len(new_slots)},
    )


def restore_synthetic_slot_store_from_previous_state(
    sandbox_after: SyntheticSharedSlotStoreSandbox,
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> SyntheticSharedSlotStoreSandbox:
    """Return a new sandbox with each slot restored to its previous vector."""
    validation = ValidationResult()
    validation.merge(sandbox_after.validation)
    restored_slots = []
    traces = list(sandbox_after.trace_records)
    for slot in sandbox_after.slots:
        trace = trace_write_prep("qdt_synthetic_slot_store.restore_slot", validation, {
            "slot_id": slot.slot_id,
            "rollback_replay": True,
            "live_store_mutated": False,
        })
        traces.append(trace)
        restored_slots.append(SyntheticSlotRecord(
            slot_id=slot.slot_id,
            canonical_slot_id=slot.canonical_slot_id,
            current_vector=slot.previous_vector,
            current_fingerprint=slot.previous_fingerprint,
            previous_vector=slot.previous_vector,
            previous_fingerprint=slot.previous_fingerprint,
            version=slot.version + 1,
            trace_id=trace.trace_id,
            metadata={**dict(slot.metadata or {}), "rollback_replayed": True, "live_store_mutated": False},
        ))
    final_trace = trace_write_prep("qdt_synthetic_slot_store.restore_synthetic_slot_store_from_previous_state", validation, {
        "slot_count": len(restored_slots),
        "live_store_mutated": False,
    })
    traces.append(final_trace)
    return SyntheticSharedSlotStoreSandbox(
        sandbox_id=write_prep3_result_id("synthetic_slot_store_restored", sandbox_after.sandbox_id),
        slots=tuple(sorted(restored_slots, key=lambda s: (s.slot_id, s.canonical_slot_id))),
        validation=validation,
        trace_records=tuple(traces),
        metadata={"synthetic_only": True, "rollback_replayed": True, "live_store_mutated": False, "slot_count": len(restored_slots)},
    )
