from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .wm_external_memory_interfaces import (
    ExternalMemoryQuery,
    ExternalMemoryResponse,
    SyntheticExternalMemoryBank,
)


class TripleHybridLTMExternalMemoryBank:
    """Production LTM adapter over EnhancedTripleHybridMemory.

    Implements the same query/response contract as SyntheticExternalMemoryBank
    but reads HG, CGMN, Curved, and fused triple-hybrid outputs from the live LTM.
    """

    _FUSED_BANK_KEY = "fused"
    _BANK_KEYS: Tuple[str, ...] = (
        "hg_episodic",
        "cgmn_semantic",
        "curved_associative",
        "procedural_spcp",
        "spatial_topological",
        "fused",
    )

    @classmethod
    def _select_bank_indices(cls, scores: torch.Tensor, top_k: int) -> List[int]:
        """Pick up to top_k banks, always retaining the holistically-fused read."""
        n_banks = len(cls._BANK_KEYS)
        k = max(1, min(int(top_k), n_banks))
        if k >= n_banks:
            return list(range(n_banks))

        try:
            fused_idx = cls._BANK_KEYS.index(cls._FUSED_BANK_KEY)
        except ValueError:
            fused_idx = None

        if fused_idx is None:
            mean_scores = scores.mean(dim=0)
            _, pick = torch.topk(mean_scores, k=k)
            return sorted(int(i) for i in pick.tolist())

        component_indices = [i for i in range(n_banks) if i != fused_idx]
        if k == 1:
            return [fused_idx]

        mean_scores = scores.mean(dim=0)
        comp_scores = mean_scores[component_indices]
        component_k = min(k - 1, len(component_indices))
        _, top_comp = torch.topk(comp_scores, k=component_k)
        selected = [component_indices[int(i)] for i in top_comp.tolist()]
        selected.append(fused_idx)
        return sorted(selected)

    @staticmethod
    def _gather_banks(
        memory_state: torch.Tensor,
        scores: torch.Tensor,
        slot_ids: List[List[str]],
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        idx = torch.tensor(indices, device=memory_state.device, dtype=torch.long)
        gathered_state = memory_state.index_select(1, idx)
        gathered_scores = scores.index_select(1, idx)
        gathered_ids = [[row[i] for i in indices] for row in slot_ids]
        return gathered_state, gathered_scores, gathered_ids

    def __init__(
        self,
        dim: int,
        triple_hybrid: Any = None,
        shared_slot_store: Any = None,
        *,
        fallback_slots: int = 8,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.memory_type = "ltm"
        self.dim = int(dim)
        self.triple_hybrid = triple_hybrid
        self.shared_slot_store = shared_slot_store
        self._fallback = SyntheticExternalMemoryBank(
            "ltm", self.dim, slots=max(fallback_slots, 8), shared_slot_store=shared_slot_store
        )
        self.last_query_trace: Dict[str, Any] = {}

    def attach(self, triple_hybrid: Any, shared_slot_store: Any = None) -> None:
        self.triple_hybrid = triple_hybrid
        if shared_slot_store is not None:
            self.shared_slot_store = shared_slot_store
            self._fallback.shared_slot_store = shared_slot_store

    @property
    def is_attached(self) -> bool:
        return self.triple_hybrid is not None

    def _wm_context_from_query(self, query: ExternalMemoryQuery) -> Optional[torch.Tensor]:
        if query.context is not None and query.context.dim() == 3:
            return query.context.detach()
        if query.depth_state is not None:
            ds = query.depth_state
            if ds.dim() == 5:
                bsz, depths, seq, triplet, dim = ds.shape
                return ds.reshape(bsz, depths * seq * triplet, dim).detach()
        return None

    def _query_sequence(self, query: ExternalMemoryQuery) -> torch.Tensor:
        if query.context is not None and query.context.dim() == 3:
            return query.context
        return query.query_state.unsqueeze(1)

    def _score_candidates(self, query_state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        qn = F.normalize(query_state, dim=-1)
        cn = F.normalize(candidates, dim=-1)
        return torch.einsum("bd,bkd->bk", qn, cn)

    def _read_kwargs_from_query(self, query: ExternalMemoryQuery) -> Dict[str, Any]:
        meta = query.metadata or {}
        out: Dict[str, Any] = {}
        fire_mask = meta.get("fire_mask")
        if fire_mask is not None:
            out["fire_mask"] = fire_mask
        if "recall_boost" in meta:
            out["recall_boost"] = float(meta["recall_boost"])
        return out

    def _read_live_candidates(
        self,
        ltm: Any,
        seq_tokens: torch.Tensor,
        query: ExternalMemoryQuery,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        wm_ctx = self._wm_context_from_query(query)
        if wm_ctx is not None and hasattr(ltm, "set_external_attention_context"):
            ltm.set_external_attention_context(wm_ctx)

        read_kwargs = self._read_kwargs_from_query(query)
        names = list(self._BANK_KEYS)
        if hasattr(ltm, "read_banks"):
            bank_reads = ltm.read_banks(seq_tokens, **read_kwargs, include_fused=True)
            reads = [
                bank_reads["hg"],
                bank_reads["cgmn"],
                bank_reads["curved"],
                bank_reads.get("spcp", bank_reads["curved"]),
                bank_reads.get("spatial", bank_reads["curved"]),
                bank_reads["fused"],
            ]
        elif hasattr(ltm, "read_bank"):
            reads = [
                ltm.read_bank("hg_episodic", seq_tokens, **read_kwargs),
                ltm.read_bank("cgmn_semantic", seq_tokens, **read_kwargs),
                ltm.read_bank("curved_associative", seq_tokens, **read_kwargs),
                ltm.read_bank("procedural_spcp", seq_tokens, **read_kwargs),
                ltm.read_bank("spatial_topological", seq_tokens, **read_kwargs),
                ltm.read_bank("fused", seq_tokens, **read_kwargs),
            ]
        else:
            reads = [
                ltm.hg(seq_tokens, operation="read", **read_kwargs),
                ltm.cgmn(seq_tokens, operation="read", **read_kwargs),
                ltm.curved(seq_tokens, operation="read"),
                ltm.curved(seq_tokens, operation="read"),
                ltm.curved(seq_tokens, operation="read"),
                ltm(seq_tokens, operation="read", **read_kwargs),
            ]

        pooled = torch.stack([r.mean(dim=1) for r in reads], dim=1)
        scores = self._score_candidates(query.query_state, pooled)
        slot_ids = [
            [f"ltm_{name}_{batch_i:04d}" for name in names]
            for batch_i in range(pooled.size(0))
        ]
        return pooled, scores, slot_ids

    def query(self, query: ExternalMemoryQuery, top_k: int = 4) -> ExternalMemoryResponse:
        query.validate()
        ltm = self.triple_hybrid
        if ltm is None:
            response = self._fallback.query(query, top_k=top_k)
            response.metadata["adapter_kind"] = "triple_hybrid_ltm_adapter_fallback"
            return response

        seq_tokens = self._query_sequence(query)
        memory_state, scores, slot_ids = self._read_live_candidates(ltm, seq_tokens, query)

        selected = self._select_bank_indices(scores, top_k)
        k = len(selected)
        memory_state, scores, slot_ids = self._gather_banks(memory_state, scores, slot_ids, selected)
        selected_bank_keys = [self._BANK_KEYS[i] for i in selected]

        shared_refs = None
        mirror_reads = bool((query.metadata or {}).get("mirror_reads_to_shared_store", False))
        if self.shared_slot_store is not None and mirror_reads:
            shared_refs = []
            for row_ids, row_mem in zip(slot_ids, memory_state.detach()):
                ref_row = []
                for local_slot_id, content in zip(row_ids, row_mem):
                    write_result = self.shared_slot_store.write_slot(
                        memory_type="ltm",
                        local_slot_id=local_slot_id,
                        content=content,
                        owner="ltm",
                        geometry_map=query.metadata.get("geometry_map"),
                        confidence=1.0,
                        write_permission=True,
                        metadata={"source": "TripleHybridLTMExternalMemoryBank.query"},
                    )
                    ref_row.append(write_result.canonical_id)
                shared_refs.append(ref_row)

        confidence = torch.sigmoid(scores.mean(dim=-1))
        pref = getattr(ltm, "last_prefusion_specialization_stats", None) or {}
        response = ExternalMemoryResponse(
            memory_type="ltm",
            memory_state=memory_state,
            scores=scores,
            slot_ids=slot_ids,
            confidence=confidence,
            trace={
                "trace_type": "triple_hybrid_ltm_memory_response",
                "request": query.to_trace(),
                "top_k": k,
                "banks": selected_bank_keys,
                "fused_included": self._FUSED_BANK_KEY in selected_bank_keys,
                "prefusion_specialization": pref,
                "shared_slot_refs": shared_refs,
            },
            metadata={
                "adapter_kind": "triple_hybrid_ltm_adapter",
                "attached": True,
                "banks_read": selected_bank_keys,
            },
            shared_slot_refs=shared_refs,
        )
        response.validate()
        self.last_query_trace = response.trace
        return response
