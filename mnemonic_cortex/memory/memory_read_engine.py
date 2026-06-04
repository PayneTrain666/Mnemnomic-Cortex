from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .shared_slot_schema import SlotMetadata, slot_state_to_code
from .shared_slot_store import SharedSlotStore


@dataclass
class MemoryReadRequest:
    requester_system: str
    query: torch.Tensor  # [B, D]
    memory_type: Optional[str] = None
    top_k: int = 16
    allowed_states: Tuple[str, ...] = ("volatile", "provisional", "durable")
    allowed_primary_systems: Optional[List[str]] = None
    use_geometry: bool = True
    geometry_family: Optional[str] = None
    geometry_depth: Optional[int] = None


@dataclass
class MemoryReadOutput:
    slot_ids: torch.Tensor  # [B, K]
    scores: torch.Tensor  # [B, K]
    values: torch.Tensor  # [B, K, D]
    metadata: List[List[SlotMetadata | Any]]
    diagnostics: Dict[str, Any]


class MemoryReadEngine(nn.Module):
    """
    Read pipeline for shared-slot memory retrieval.

    Important shapes:
      - global slot values: [N, D]
      - candidate values: [C, D]
      - batched candidate values: [B, C, D]
      - output values: [B, K, D]
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        geometry_runtime: Any = None,
        reranker: Any = None,
    ) -> None:
        super().__init__()
        self.store = store  # shared slot substrate
        self.geometry_runtime = geometry_runtime  # geometry runtime dependency
        self.reranker = reranker  # retrieval reranker transformer dependency

    def filter_candidate_slots(self, request: MemoryReadRequest) -> torch.Tensor:
        # Returns candidate slot ids [C] on store device.
        allowed_state_codes = {slot_state_to_code(s) for s in request.allowed_states}
        ids: List[int] = []
        for slot_id in range(self.store.num_slots):
            state_code = int(self.store.slot_state_code[slot_id].item())
            if state_code not in allowed_state_codes:
                continue
            meta = self.store.metadata.get(slot_id)
            if request.allowed_primary_systems is not None and meta is not None:
                primary = getattr(meta, "primary_system_id", None)
                if primary not in request.allowed_primary_systems:
                    continue
            ids.append(slot_id)

        return torch.as_tensor(ids, device=self.store.slot_values.device, dtype=torch.long)

    def score_candidates_euclidean(self, query: torch.Tensor, candidate_values: torch.Tensor) -> torch.Tensor:
        """
        query: [B, D]
        candidate_values: [C, D] or [B, C, D]
        returns scores [B, C] (higher is better).
        """
        if candidate_values.dim() == 2:
            # [C, D] -> [B, C, D]
            candidate_values = candidate_values.unsqueeze(0).expand(query.size(0), -1, -1)
        if candidate_values.dim() != 3:
            raise ValueError("candidate_values must be [C, D] or [B, C, D]")
        if query.dim() != 2:
            raise ValueError("query must be [B, D]")

        # Negative squared Euclidean distance.
        delta = candidate_values - query.unsqueeze(1)
        return -torch.sum(delta * delta, dim=-1)

    def score_candidates_geometry(
        self,
        *,
        request: MemoryReadRequest,
        query: torch.Tensor,
        candidate_values: torch.Tensor,  # [C, D] or [B, C, D]
    ) -> torch.Tensor:
        if self.geometry_runtime is None:
            return self.score_candidates_euclidean(query, candidate_values)

        if hasattr(self.geometry_runtime, "score"):
            scored = self.geometry_runtime.score(
                query=query,
                candidate_values=candidate_values,
                family=request.geometry_family,
                depth=request.geometry_depth,
            )
        elif callable(self.geometry_runtime):
            scored = self.geometry_runtime(
                query=query,
                candidate_values=candidate_values,
                family=request.geometry_family,
                depth=request.geometry_depth,
            )
        else:
            raise TypeError("geometry_runtime must be callable or expose .score(...)")

        scores = torch.as_tensor(scored, device=query.device, dtype=query.dtype)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0).expand(query.size(0), -1)
        if scores.dim() != 2:
            raise ValueError("geometry scores must resolve to [B, C]")
        return scores

    def rerank(
        self,
        *,
        query: torch.Tensor,
        candidate_values: torch.Tensor,  # [B, C, D]
        candidate_scores: torch.Tensor,  # [B, C]
    ) -> torch.Tensor:
        if self.reranker is None:
            return candidate_scores

        if hasattr(self.reranker, "forward"):
            out = self.reranker(query, candidate_values, candidate_scores)
        elif callable(self.reranker):
            out = self.reranker(query, candidate_values, candidate_scores)
        else:
            raise TypeError("reranker must be callable or nn.Module-like")

        reranked = torch.as_tensor(out, device=query.device, dtype=query.dtype)
        if reranked.dim() != 2 or reranked.shape != candidate_scores.shape:
            raise ValueError("reranker output must be [B, C]")
        return reranked

    def retrieve(self, request: MemoryReadRequest) -> MemoryReadOutput:
        query = request.query.to(self.store.slot_values.device, dtype=self.store.slot_values.dtype)
        if query.dim() != 2:
            raise ValueError("request.query must be [B, D]")
        bsz, d = query.shape
        if d != self.store.slot_dim:
            raise ValueError(f"query D={d} does not match slot_dim={self.store.slot_dim}")

        candidate_slot_ids = self.filter_candidate_slots(request)  # [C]
        c = int(candidate_slot_ids.numel())
        if c == 0:
            empty_ids = torch.zeros(bsz, 0, device=query.device, dtype=torch.long)
            empty_scores = torch.zeros(bsz, 0, device=query.device, dtype=query.dtype)
            empty_vals = torch.zeros(bsz, 0, d, device=query.device, dtype=query.dtype)
            return MemoryReadOutput(
                slot_ids=empty_ids,
                scores=empty_scores,
                values=empty_vals,
                metadata=[[] for _ in range(bsz)],
                diagnostics={"candidate_count": 0, "top_k": 0},
            )

        # global slot values [N, D] -> candidate values [C, D]
        candidate_values = self.store.slot_values.index_select(0, candidate_slot_ids)
        # batched candidate values [B, C, D]
        batched_candidate_values = candidate_values.unsqueeze(0).expand(bsz, -1, -1)

        if request.use_geometry:
            candidate_scores = self.score_candidates_geometry(
                request=request,
                query=query,
                candidate_values=batched_candidate_values,
            )
        else:
            candidate_scores = self.score_candidates_euclidean(query, batched_candidate_values)

        candidate_scores = self.rerank(
            query=query,
            candidate_values=batched_candidate_values,
            candidate_scores=candidate_scores,
        )

        k = min(max(1, int(request.top_k)), c)
        top_scores, top_indices = torch.topk(candidate_scores, k=k, dim=1)
        top_slot_ids = candidate_slot_ids.index_select(0, top_indices.reshape(-1)).reshape(bsz, k)

        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, d)
        out_values = batched_candidate_values.gather(1, gather_idx)  # [B, K, D]

        out_metadata: List[List[SlotMetadata | Any]] = []
        for batch_row in top_slot_ids.tolist():
            out_metadata.append([self.store.metadata.get(int(slot_id)) for slot_id in batch_row])

        diagnostics = {
            "candidate_count": c,
            "top_k": k,
            "used_geometry": bool(request.use_geometry and self.geometry_runtime is not None),
            "used_reranker": self.reranker is not None,
            "global_shape": list(self.store.slot_values.shape),  # [N, D]
            "candidate_shape": list(candidate_values.shape),  # [C, D]
            "batched_candidate_shape": list(batched_candidate_values.shape),  # [B, C, D]
            "output_shape": list(out_values.shape),  # [B, K, D]
        }
        return MemoryReadOutput(
            slot_ids=top_slot_ids,
            scores=top_scores,
            values=out_values,
            metadata=out_metadata,
            diagnostics=diagnostics,
        )
