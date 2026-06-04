from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..memory.memory_lifecycle_manager import MemoryLifecycleManager
from ..memory.memory_read_engine import MemoryReadEngine, MemoryReadOutput, MemoryReadRequest
from ..memory.memory_update_engine import MemoryUpdateEngine
from ..memory.memory_write_engine import MemoryWriteEngine
from ..memory.shared_slot_schema import SlotWriteRequest
from ..memory.shared_slot_store import SharedSlotStore


@dataclass
class EpisodeRecord:
    episode_id: str
    slot_ids: List[int]
    start_step: int
    end_step: int
    anchor_time: Optional[float]
    trace_ids: List[str]
    causal_links: List[str] = field(default_factory=list)
    summary_slot_ids: List[int] = field(default_factory=list)
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)


class HGEpisodicLTM(nn.Module):
    """
    Episodic LTM wrapper on shared-slot memory stack.

    Storage doctrine:
      - episode vectors are first written as provisional.
      - lifecycle manager may later promote highly-used vectors to durable.
      - long episodes get additional summary slots.

    Retrieval doctrine defaults:
      - requester_system = "hg_ep_ltm"
      - memory_type = "episodic"
      - geometry enabled
      - family/depth are resolved from optional geometry policy runtime.
    """

    def __init__(
        self,
        *,
        slot_store: SharedSlotStore,
        read_engine: MemoryReadEngine,
        write_engine: MemoryWriteEngine,
        update_engine: MemoryUpdateEngine,
        lifecycle: MemoryLifecycleManager,
        slot_dim: int,
        geometry_policy_runtime: Any = None,
        long_episode_threshold: int = 16,
        summary_stride: int = 8,
        promotion_retrieval_threshold: int = 3,
    ) -> None:
        super().__init__()
        self.slot_store = slot_store
        self.read_engine = read_engine
        self.write_engine = write_engine
        self.update_engine = update_engine
        self.lifecycle = lifecycle
        self.slot_dim = int(slot_dim)
        self.geometry_policy_runtime = geometry_policy_runtime
        self.long_episode_threshold = int(max(2, long_episode_threshold))
        self.summary_stride = int(max(2, summary_stride))
        self.promotion_retrieval_threshold = int(max(1, promotion_retrieval_threshold))

        self.episode_records: Dict[str, EpisodeRecord] = {}
        self.episode_index_by_trace_id: Dict[str, List[str]] = {}
        self.episode_index_by_time: List[Tuple[int, int, str]] = []
        self.episode_index_by_tag: Dict[str, List[str]] = {}
        self.slot_retrieval_hits: Dict[int, int] = {}

    def _resolve_geometry_policy(self) -> Tuple[Optional[str], Optional[int]]:
        runtime = self.geometry_policy_runtime
        if runtime is None:
            return None, None
        if hasattr(runtime, "resolve_family_depth"):
            out = runtime.resolve_family_depth(memory_type="episodic")
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
        if hasattr(runtime, "family") or hasattr(runtime, "depth"):
            return getattr(runtime, "family", None), getattr(runtime, "depth", None)
        return None, None

    def _index_episode(self, record: EpisodeRecord) -> None:
        self.episode_records[record.episode_id] = record
        self.episode_index_by_time.append((record.start_step, record.end_step, record.episode_id))
        for trace_id in record.trace_ids:
            self.episode_index_by_trace_id.setdefault(trace_id, []).append(record.episode_id)
        for tag in record.tags:
            self.episode_index_by_tag.setdefault(tag, []).append(record.episode_id)

    def _episode_write_request(
        self,
        *,
        vectors: torch.Tensor,
        confidence: float,
        trace_ids: List[str],
        tags: List[str],
        extra: Dict[str, Any],
    ) -> SlotWriteRequest:
        return SlotWriteRequest(
            requester_system="hg_ep_ltm",
            candidate_value_shape=list(vectors.shape),
            requested_state="provisional",
            requested_memory_type="episodic",
            confidence=float(confidence),
            semantic_tags=list(tags),
            provenance_trace_ids=list(trace_ids),
            extra=dict(extra),
        )

    def store_episode(
        self,
        *,
        episode_id: str,
        episode_vectors: torch.Tensor,  # [T, D]
        step_range: Tuple[int, int],
        anchor_time: Optional[float] = None,
        trace_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> EpisodeRecord:
        vectors = torch.as_tensor(
            episode_vectors,
            device=self.slot_store.slot_values.device,
            dtype=self.slot_store.slot_values.dtype,
        )
        if vectors.dim() != 2 or vectors.size(1) != self.slot_dim:
            raise ValueError(f"episode_vectors must be [T, D={self.slot_dim}]")
        if not torch.isfinite(vectors).all():
            raise ValueError("episode_vectors contains NaN or Inf")

        start_step, end_step = int(step_range[0]), int(step_range[1])
        traces = list(trace_ids or [f"episode:{episode_id}"])
        ep_tags = list(tags or [])
        base_extra = {
            "episode_id": episode_id,
            "step_range": [start_step, end_step],
            "anchor_time": anchor_time,
            "storage_doctrine": "provisional_first",
        }

        write_request = self._episode_write_request(
            vectors=vectors,
            confidence=0.80,
            trace_ids=traces,
            tags=ep_tags + [f"episode:{episode_id}"],
            extra=base_extra,
        )
        write_out = self.write_engine.write(request=write_request, values=vectors)
        slot_ids = [int(x) for x in write_out.written_slot_ids]

        summary_slot_ids: List[int] = []
        if vectors.size(0) >= self.long_episode_threshold:
            # Long episodes get summary slots (global + chunk summaries).
            global_summary = vectors.mean(dim=0, keepdim=True)
            chunk_summaries = [
                vectors[i : i + self.summary_stride].mean(dim=0, keepdim=True)
                for i in range(0, vectors.size(0), self.summary_stride)
            ]
            summary_values = torch.cat([global_summary] + chunk_summaries, dim=0)
            summary_request = self._episode_write_request(
                vectors=summary_values,
                confidence=0.75,
                trace_ids=traces,
                tags=ep_tags + [f"episode:{episode_id}", "episode_summary"],
                extra={**base_extra, "is_summary": True},
            )
            summary_out = self.write_engine.write(request=summary_request, values=summary_values)
            summary_slot_ids = [int(x) for x in summary_out.written_slot_ids]

        record = EpisodeRecord(
            episode_id=episode_id,
            slot_ids=slot_ids,
            start_step=start_step,
            end_step=end_step,
            anchor_time=anchor_time,
            trace_ids=traces,
            summary_slot_ids=summary_slot_ids,
            confidence=float(write_out.confidence),
            tags=ep_tags,
        )
        self._index_episode(record)
        return record

    def _passes_optional_filters(
        self,
        *,
        slot_meta: Any,
        tags: Optional[List[str]],
        time_window: Optional[Tuple[int, int]],
    ) -> bool:
        if slot_meta is None:
            return False
        extra = getattr(slot_meta, "extra", {}) or {}
        if tags:
            existing_tags = set(getattr(slot_meta, "semantic_tags", []) or [])
            if set(tags).isdisjoint(existing_tags):
                return False
        if time_window:
            start_end = extra.get("step_range")
            if not isinstance(start_end, list) or len(start_end) != 2:
                return False
            req_start, req_end = int(time_window[0]), int(time_window[1])
            slot_start, slot_end = int(start_end[0]), int(start_end[1])
            overlaps = not (slot_end < req_start or slot_start > req_end)
            if not overlaps:
                return False
        return True

    def _track_retrieval_use(self, slot_ids: torch.Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        flat_ids = [int(x) for x in slot_ids.reshape(-1).tolist()]
        if flat_ids:
            self.slot_store.increment_usage(flat_ids, amount=1.0)
        unique_ids = sorted(set(flat_ids))
        for sid in unique_ids:
            hits = int(self.slot_retrieval_hits.get(sid, 0)) + 1
            self.slot_retrieval_hits[sid] = hits
            if hits >= self.promotion_retrieval_threshold:
                decision = self.lifecycle.evaluate_promotion(sid)
                if decision.new_state != decision.old_state:
                    self.lifecycle.apply_decision(decision)

    def retrieve_episode_fragments(
        self,
        *,
        query: torch.Tensor,  # [B, D]
        top_k: int = 16,
        tags: Optional[List[str]] = None,
        time_window: Optional[Tuple[int, int]] = None,
    ) -> MemoryReadOutput:
        family, depth = self._resolve_geometry_policy()
        request = MemoryReadRequest(
            requester_system="hg_ep_ltm",
            query=query,
            memory_type="episodic",
            top_k=int(top_k),
            use_geometry=True,
            geometry_family=family,
            geometry_depth=depth,
            allowed_primary_systems=["hg_ep_ltm"],
        )
        out = self.read_engine.retrieve(request)

        if tags or time_window:
            filtered_slot_rows: List[torch.Tensor] = []
            filtered_score_rows: List[torch.Tensor] = []
            filtered_value_rows: List[torch.Tensor] = []
            filtered_meta_rows: List[List[Any]] = []
            for b in range(out.slot_ids.size(0)):
                keep: List[int] = []
                for i, meta in enumerate(out.metadata[b]):
                    if self._passes_optional_filters(slot_meta=meta, tags=tags, time_window=time_window):
                        keep.append(i)
                if not keep:
                    keep = list(range(out.slot_ids.size(1)))
                idx = torch.as_tensor(keep, device=out.slot_ids.device, dtype=torch.long)
                filtered_slot_rows.append(out.slot_ids[b].index_select(0, idx))
                filtered_score_rows.append(out.scores[b].index_select(0, idx))
                filtered_value_rows.append(out.values[b].index_select(0, idx))
                filtered_meta_rows.append([out.metadata[b][int(i)] for i in keep])

            k = min(x.numel() for x in filtered_slot_rows) if filtered_slot_rows else 0
            if k > 0:
                slot_ids = torch.stack([x[:k] for x in filtered_slot_rows], dim=0)
                scores = torch.stack([x[:k] for x in filtered_score_rows], dim=0)
                values = torch.stack([x[:k] for x in filtered_value_rows], dim=0)
                metadata = [x[:k] for x in filtered_meta_rows]
                out = MemoryReadOutput(
                    slot_ids=slot_ids,
                    scores=scores,
                    values=values,
                    metadata=metadata,
                    diagnostics={**dict(out.diagnostics), "post_filter_top_k": int(k)},
                )

        self._track_retrieval_use(out.slot_ids)
        return out

    def stitch_related_episodes(
        self,
        *,
        episode_id: str,
        max_hops: int = 3,
    ) -> List[str]:
        if episode_id not in self.episode_records:
            return []
        visited: Set[str] = set()
        frontier: Set[str] = {episode_id}
        hops = max(0, int(max_hops))
        for _ in range(hops + 1):
            next_frontier: Set[str] = set()
            for eid in frontier:
                if eid in visited:
                    continue
                visited.add(eid)
                rec = self.episode_records.get(eid)
                if rec is None:
                    continue
                for linked in rec.causal_links:
                    if linked not in visited:
                        next_frontier.add(linked)
                for tr in rec.trace_ids:
                    for linked in self.episode_index_by_trace_id.get(tr, []):
                        if linked not in visited:
                            next_frontier.add(linked)
                for tag in rec.tags:
                    for linked in self.episode_index_by_tag.get(tag, []):
                        if linked not in visited:
                            next_frontier.add(linked)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(visited)

    def summarize_episode(
        self,
        *,
        episode_id: str,
    ) -> List[int]:
        record = self.episode_records.get(episode_id)
        if record is None:
            return []
        if record.summary_slot_ids:
            return list(record.summary_slot_ids)

        vectors = self.slot_store.get_slot_value(record.slot_ids)
        if vectors.numel() == 0:
            return []
        summary_value = vectors.mean(dim=0, keepdim=True)
        summary_request = self._episode_write_request(
            vectors=summary_value,
            confidence=0.70,
            trace_ids=list(record.trace_ids),
            tags=list(record.tags) + [f"episode:{episode_id}", "episode_summary"],
            extra={
                "episode_id": episode_id,
                "step_range": [record.start_step, record.end_step],
                "anchor_time": record.anchor_time,
                "is_summary": True,
            },
        )
        out = self.write_engine.write(request=summary_request, values=summary_value)
        record.summary_slot_ids = [int(x) for x in out.written_slot_ids]
        return list(record.summary_slot_ids)
