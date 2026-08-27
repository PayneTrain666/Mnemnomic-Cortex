"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm external memory interfaces.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

from .wm_external_memory_guards import ensure_external_memory_response, ensure_mann_trace_visibility, ensure_fusion_inputs, ensure_shared_slot_id, ensure_shared_slot_record, ensure_qh_code_schema, ensure_qh_storage_record, interference_score, external_memory_contract_trace, external_memory_trace

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid

import torch

from geometry.chart_native import majority_chart, mix_chart_affinity

from .wm_shared_slot_store import SharedSlotStore, SharedSlotStoreConfig


MEMORY_TYPES = ("ltm", "mann", "spcp")


@dataclass
class ExternalMemoryQuery:
    """Query object sent from WM to an external memory subsystem.

    Contract:
    - query_state: [B,D]
    - optional depth_state: [B,Z,T,3,D]
    - memory_type: one of ltm/mann/spcp
    """

    memory_type: str
    query_state: torch.Tensor
    depth_state: Optional[torch.Tensor] = None
    context: Optional[torch.Tensor] = None
    request_id: str = field(default_factory=lambda: f"wmq-{uuid.uuid4().hex}")
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_slot_refs: Optional[List[List[str]]] = None

    def validate(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}")
        if self.query_state.dim() != 2:
            raise ValueError("query_state must be [B,D]")
        if not torch.isfinite(self.query_state).all():
            self.query_state = torch.nan_to_num(self.query_state, nan=0.0, posinf=0.0, neginf=0.0)
        if self.depth_state is not None:
            if self.depth_state.dim() != 5 or self.depth_state.size(-2) != 3:
                raise ValueError("depth_state must be [B,Z,T,3,D]")
            if self.depth_state.size(0) != self.query_state.size(0):
                raise ValueError("depth_state batch must match query_state")
            if not torch.isfinite(self.depth_state).all():
                self.depth_state = torch.nan_to_num(self.depth_state, nan=0.0, posinf=0.0, neginf=0.0)
        if self.context is not None:
            if self.context.size(0) != self.query_state.size(0):
                raise ValueError("context batch must match query_state")
            if not torch.isfinite(self.context).all():
                self.context = torch.nan_to_num(self.context, nan=0.0, posinf=0.0, neginf=0.0)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "memory_type": self.memory_type,
            "query_shape": list(self.query_state.shape),
            "depth_shape": None if self.depth_state is None else list(self.depth_state.shape),
            "context_shape": None if self.context is None else list(self.context.shape),
            "metadata": self.metadata,
            "shared_slot_refs": self.shared_slot_refs,
        }


@dataclass
class ExternalMemoryResponse:
    """Normalized response object returned from external memory to WM.

    Contract:
    - memory_state: [B,K,D]
    - scores: [B,K]
    """

    memory_type: str
    memory_state: torch.Tensor
    scores: torch.Tensor
    slot_ids: List[List[str]]
    confidence: torch.Tensor
    trace: Dict[str, Any] = field(default_factory=dict)
    scratchpad_tokens: Optional[torch.Tensor] = None
    per_hop_attention: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_slot_refs: Optional[List[List[str]]] = None

    def validate(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}")
        if self.memory_state.dim() != 3:
            raise ValueError("memory_state must be [B,K,D]")
        if self.scores.dim() != 2:
            raise ValueError("scores must be [B,K]")
        if self.memory_state.shape[:2] != self.scores.shape:
            raise ValueError("memory_state [B,K] must match scores [B,K]")
        if self.confidence.dim() != 1 or self.confidence.size(0) != self.memory_state.size(0):
            raise ValueError("confidence must be [B]")
        if len(self.slot_ids) != self.memory_state.size(0):
            raise ValueError("slot_ids outer list must match batch")
        if any(len(row) != self.memory_state.size(1) for row in self.slot_ids):
            raise ValueError("slot_ids inner lists must match K")
        if not torch.isfinite(self.memory_state).all():
            self.memory_state = torch.nan_to_num(self.memory_state, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(self.scores).all():
            self.scores = torch.nan_to_num(self.scores, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(self.confidence).all():
            self.confidence = torch.nan_to_num(self.confidence, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scratchpad_tokens is not None:
            if self.scratchpad_tokens.size(0) != self.memory_state.size(0) or self.scratchpad_tokens.size(-1) != self.memory_state.size(-1):
                raise ValueError("scratchpad_tokens must be [B,H,D] with matching B,D")
            if not torch.isfinite(self.scratchpad_tokens).all():
                self.scratchpad_tokens = torch.nan_to_num(
                    self.scratchpad_tokens, nan=0.0, posinf=0.0, neginf=0.0
                )
        if self.per_hop_attention is not None:
            if self.per_hop_attention.size(0) != self.memory_state.size(0):
                raise ValueError("per_hop_attention batch must match")
            if not torch.isfinite(self.per_hop_attention).all():
                self.per_hop_attention = torch.nan_to_num(
                    self.per_hop_attention, nan=0.0, posinf=0.0, neginf=0.0
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_type": self.memory_type,
            "memory_state_shape": list(self.memory_state.shape),
            "scores_shape": list(self.scores.shape),
            "slot_ids": self.slot_ids,
            "confidence": self.confidence.detach().cpu().tolist(),
            "scratchpad_tokens_shape": None if self.scratchpad_tokens is None else list(self.scratchpad_tokens.shape),
            "per_hop_attention_shape": None if self.per_hop_attention is None else list(self.per_hop_attention.shape),
            "trace": self.trace,
            "metadata": self.metadata,
            "shared_slot_refs": self.shared_slot_refs,
        }


class SyntheticExternalMemoryBank:
    """Deterministic in-process external memory adapter for tests and local integration.

    This is a practical interface shim, not a claim that real LTM/MANN/SPCP
    storage has been implemented. Real subsystem adapters can later implement
    the same query/response contract.
    """

    def __init__(self, memory_type: str, dim: int, slots: int = 8, hops: int = 3, shared_slot_store: Optional[SharedSlotStore] = None):
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}")
        if dim <= 0 or slots <= 0 or hops <= 0:
            raise ValueError("dim, slots, and hops must be positive")
        self.memory_type = memory_type
        self.dim = dim
        self.slots = slots
        self.hops = hops
        base = torch.linspace(-1.0, 1.0, steps=slots * dim, dtype=torch.float32).reshape(slots, dim)
        if memory_type == "ltm":
            self.prototype = torch.tanh(base)
        elif memory_type == "mann":
            self.prototype = torch.sin(base * 3.14159)
        else:
            self.prototype = torch.cos(base * 1.57079)
        self.slot_ids = [f"{memory_type}_slot_{i:04d}" for i in range(slots)]
        self.shared_slot_store = shared_slot_store

    def query(self, query: ExternalMemoryQuery, top_k: int = 4) -> ExternalMemoryResponse:
        query.validate()
        q = query.query_state
        proto = self.prototype.to(device=q.device, dtype=q.dtype)
        qn = torch.nn.functional.normalize(q, dim=-1)
        pn = torch.nn.functional.normalize(proto, dim=-1)
        scores_full = torch.matmul(qn, pn.t())
        meta = query.metadata or {}
        chart_mix = float(meta.get("native_chart_mix", 0.0) or 0.0)
        if chart_mix > 0.0:
            charts = meta.get("geometry_by_depth")
            geometry = str(meta.get("geometry") or majority_chart(charts, default="euclidean"))
            scores_full = mix_chart_affinity(scores_full, q, proto, geometry, chart_mix)
        k = min(top_k, self.slots)
        scores, idx = torch.topk(scores_full, k=k, dim=-1)
        memory = proto.index_select(0, idx.reshape(-1)).reshape(q.size(0), k, self.dim)
        ids = [[self.slot_ids[int(i)] for i in row] for row in idx.detach().cpu().tolist()]
        shared_refs = None
        if self.shared_slot_store is not None:
            shared_refs = []
            for batch_i, row in enumerate(idx.detach().cpu().tolist()):
                ref_row = []
                for local_idx in row:
                    local_slot_id = self.slot_ids[int(local_idx)]
                    content = proto[int(local_idx)].detach()
                    write_result = self.shared_slot_store.write_slot(
                        memory_type=self.memory_type,
                        local_slot_id=local_slot_id,
                        content=content,
                        owner="shared",
                        geometry_map=query.metadata.get("geometry_map"),
                        confidence=1.0,
                        write_permission=True,
                        metadata={"source": "SyntheticExternalMemoryBank.query"},
                    )
                    ref_row.append(write_result.canonical_id)
                shared_refs.append(ref_row)
        confidence = torch.sigmoid(scores.mean(dim=-1))
        scratchpad = None
        per_hop = None
        if self.memory_type == "mann":
            # [B,H,D] scratchpad and [B,H,K] per-hop attention visibility.
            hop_scales = torch.linspace(0.25, 1.0, steps=self.hops, device=q.device, dtype=q.dtype).view(1, self.hops, 1)
            scratchpad = q.unsqueeze(1) * hop_scales
            hop_logits = scores.unsqueeze(1) * hop_scales
            per_hop = torch.softmax(hop_logits, dim=-1)
        response = ExternalMemoryResponse(
            memory_type=self.memory_type,
            memory_state=memory,
            scores=scores,
            slot_ids=ids,
            confidence=confidence,
            scratchpad_tokens=scratchpad,
            per_hop_attention=per_hop,
            trace={
                "trace_type": f"synthetic_{self.memory_type}_memory_response",
                "request": query.to_trace(),
                "top_k": k,
                "confidence": confidence.detach().cpu().tolist(),
                "shared_slot_refs": shared_refs,
            },
            metadata={
                "adapter_kind": "synthetic_contract_adapter",
                "real_external_adapter_required_for_production": True,
            },
            shared_slot_refs=shared_refs,
        )
        response.validate()
        return response


# ---------------------------------------------------------------------------
# WM-QD-4A external-memory/shared-slot/QH quality contract
# ---------------------------------------------------------------------------

def wm_qd4a_external_memory_contract() -> dict:
    """Return serialization-safe quality metadata for this external-memory layer.

    This no-mutation contract declares external memory response schemas, MANN
    trace visibility, fusion shape checks, shared-slot ownership/conflict
    metadata, QH code schema validation, interference checks, trace
    serialization, PAAMA-X write-permission metadata, fallback behavior, and
    compatibility with QDTWorkingMemory.
    """
    return external_memory_contract_trace(module=__name__)
