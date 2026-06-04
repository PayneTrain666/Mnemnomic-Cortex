"""Read-only proposal materialization contracts for HGM->QDT/WM.

The functions here produce finite bounded tensor *previews* as tuples of
floats.  They do not construct SystemWriteProposal instances or stage writes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

from .hgm4_result import HGMBridgePayload, TraceSafeMemoryPlan
from .hgm_qdt_write_prep_result import (
    HGMQDTWritePrepOptions,
    ProposalMaterializationContract,
    TensorProposalPreview,
    trace_write_prep,
    write_prep_stable_hash,
)
from .validation import ValidationResult
from .qdt_write_contract_probe import _coerce_options


def _fingerprint(values: Sequence[float]) -> str:
    payload = ",".join(f"{float(v):.6f}" for v in values[:64]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _payloads_from_input(payloads_or_plan: Any) -> tuple[HGMBridgePayload, ...]:
    if isinstance(payloads_or_plan, TraceSafeMemoryPlan):
        return tuple(payloads_or_plan.bridge_payloads or tuple())
    if isinstance(payloads_or_plan, HGMBridgePayload):
        return (payloads_or_plan,)
    if isinstance(payloads_or_plan, Iterable) and not isinstance(payloads_or_plan, (str, bytes, Mapping)):
        return tuple(p for p in payloads_or_plan if isinstance(p, HGMBridgePayload))
    return tuple()


def _vector_from_payload(payload: HGMBridgePayload, dim: int) -> tuple[float, ...]:
    seed = f"{payload.payload_id}|{payload.source_type}|{payload.source_id}|{payload.depth_layer.value}|{payload.geometry_type.value}|{payload.content_summary}|{payload.qspin_signature_id}"
    raw = hashlib.sha256(seed.encode("utf-8")).digest()
    values = []
    idx = 0
    while len(values) < dim:
        if idx >= len(raw):
            raw = hashlib.sha256(raw).digest()
            idx = 0
        # deterministic bounded value in [-1, 1]
        val = (raw[idx] / 255.0) * 2.0 - 1.0
        values.append(round(float(val), 6))
        idx += 1
    return tuple(values)


def build_proposal_materialization_contract(payloads_or_plan: Any, config=None, options: Optional[HGMQDTWritePrepOptions | Mapping[str, Any]] = None) -> ProposalMaterializationContract:
    opts = _coerce_options(options)
    validation = ValidationResult()
    traces = []
    payloads = _payloads_from_input(payloads_or_plan)
    if not payloads:
        validation.warning("proposal_contract.empty_payloads", "no HGMBridgePayload records supplied", "payloads")
    if len(payloads) > opts.max_payloads:
        validation.warning("proposal_contract.bounded_payloads", "payload count exceeded max_payloads; previews truncated", "payloads")
    previews = []
    for payload in payloads[: opts.max_payloads]:
        vector = _vector_from_payload(payload, opts.proposal_dim)
        finite = all(math.isfinite(v) for v in vector)
        bounded = all(abs(float(v)) <= 1.0 for v in vector)
        if not finite:
            validation.error("proposal_contract.non_finite_vector", "preview vector contains non-finite values", payload.payload_id)
        if not bounded:
            validation.error("proposal_contract.unbounded_vector", "preview vector is outside bounded range", payload.payload_id)
        trace = trace_write_prep("qdt_proposal_materialization.preview", validation, {"payload_id": payload.payload_id, "dim": opts.proposal_dim, "write_permission": False})
        traces.append(trace)
        previews.append(TensorProposalPreview(
            preview_id=f"tensor_preview_{write_prep_stable_hash(payload.payload_id, opts.proposal_dim)}",
            source_payload_id=payload.payload_id,
            content_vector=vector,
            content_shape=(opts.proposal_dim,),
            content_fingerprint=_fingerprint(vector),
            finite=finite,
            bounded=bounded,
            write_permission=False,
            trace_id=trace.trace_id,
            metadata={
                "materializes_to": "SystemWriteProposal.content",
                "preview_only": True,
                "no_tensor_constructed": True,
                "source_type": payload.source_type,
            },
        ))
    final_trace = trace_write_prep("qdt_proposal_materialization.build_proposal_materialization_contract", validation, {"preview_count": len(previews), "proposal_dim": opts.proposal_dim})
    traces.append(final_trace)
    return ProposalMaterializationContract(
        contract_id=f"proposal_contract_{write_prep_stable_hash(len(previews), opts.proposal_dim)}",
        tensor_previews=tuple(previews),
        proposal_dim=opts.proposal_dim,
        proposal_class_name="SystemWriteProposal",
        write_permission_default=False,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"read_only": True, "preview_only": True, "no_live_write": True},
    )
