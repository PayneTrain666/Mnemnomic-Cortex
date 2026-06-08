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
from .transformer_policy import DualTransformerPolicy


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
        transformer_layers: int = 0,
        fixed_transformer_layers: int = 3,
        fusion_transformer_layers: int = 0,
        decoder_transformer_layers: int = 0,
        inherited_bank_layers: int = 3,
        inherited_fusion_layers: int | None = None,
        cross_model_attention_layers: int = 4,
        attention_type: str = "multiscale",
        transformer_heads: int = 0,
        transformer_dropout: float = 0.1,
        wm_lattice_mirror: Any = None,
        wm_shared_slot_store_mirror: Any = None,
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
        self.attention_type = str(attention_type).strip().lower()
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_heads = int(transformer_heads) if int(transformer_heads) > 0 else self._pick_num_heads(self.slot_dim)
        self.cross_model_attention_layers = int(max(0, cross_model_attention_layers))
        self.transformer_policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=int(transformer_layers),
            fixed_layers_cfg=int(fixed_transformer_layers),
            fusion_layers_cfg=int(fusion_transformer_layers),
            decoder_layers_cfg=int(decoder_transformer_layers),
            inherited_bank_layers=int(inherited_bank_layers),
            inherited_fusion_layers=inherited_fusion_layers,
        )
        self.transformer_layers = int(self.transformer_policy.bank_layers)
        self.fixed_transformer_layers = int(self.transformer_policy.fixed_layers)
        self.fusion_transformer_layers = int(self.transformer_policy.fusion_layers)
        self.decoder_transformer_layers = int(self.transformer_policy.decoder_layers)
        self.wm_lattice_mirror = wm_lattice_mirror
        self.wm_shared_slot_store_mirror = wm_shared_slot_store_mirror
        self._build_transformer_stacks(
            bank_layers=self.transformer_layers,
            fixed_layers=self.fixed_transformer_layers,
            fusion_layers=self.fusion_transformer_layers,
            decoder_layers=self.decoder_transformer_layers,
            n_heads=self.transformer_heads,
            attention_type=self.attention_type,
        )

        self._qh_triplets_stored = 0
        self._qh_active_slots: Set[int] = set()
        self.episode_records: Dict[str, EpisodeRecord] = {}
        self.episode_index_by_trace_id: Dict[str, List[str]] = {}
        self.episode_index_by_time: List[Tuple[int, int, str]] = []
        self.episode_index_by_tag: Dict[str, List[str]] = {}
        self.slot_retrieval_hits: Dict[int, int] = {}
        self.external_attention_context = None
        self.query_context_attn = nn.MultiheadAttention(
            self.slot_dim,
            num_heads=self.transformer_heads,
            batch_first=True,
        )
        self.query_context_norm = nn.LayerNorm(self.slot_dim)

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    def _build_transformer_stack(self, n_layers: int, n_heads: int, attention_type: str) -> nn.Module:
        if int(n_layers) <= 0:
            return None
        if str(attention_type).strip().lower() == "multiscale":
            from ..memory_attention import MultiScaleAttention

            return nn.Sequential(
                *[
                    nn.Sequential(
                        MultiScaleAttention(self.slot_dim, n_heads),
                        nn.LayerNorm(self.slot_dim),
                    )
                    for _ in range(int(n_layers))
                ]
            )
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.slot_dim,
                nhead=n_heads,
                dim_feedforward=max(128, self.slot_dim * 2),
                dropout=float(self.transformer_dropout),
                activation="gelu",
                batch_first=True,
            ),
            num_layers=int(n_layers),
        )

    def _build_transformer_stacks(
        self,
        *,
        bank_layers: int,
        fixed_layers: int,
        fusion_layers: int,
        decoder_layers: int,
        n_heads: int,
        attention_type: str,
    ) -> None:
        self.transformer_layers = int(max(0, bank_layers))
        self.fixed_transformer_layers = int(max(0, fixed_layers))
        self.fusion_transformer_layers = int(max(0, fusion_layers))
        self.decoder_transformer_layers = int(max(0, decoder_layers))
        self.transformer_heads = int(n_heads)
        self.attention_type = str(attention_type)

        self.sequence_transformer = self._build_transformer_stack(
            self.transformer_layers,
            self.transformer_heads,
            self.attention_type,
        )
        self.sequence_transformer_norm = (
            nn.LayerNorm(self.slot_dim) if self.sequence_transformer is not None else None
        )
        self.aux_transformer = self._build_transformer_stack(
            self.fixed_transformer_layers,
            self.transformer_heads,
            self.attention_type,
        )
        self.aux_norm = nn.LayerNorm(self.slot_dim) if self.aux_transformer is not None else None
        self.fusion_refiner = self._build_transformer_stack(
            self.fusion_transformer_layers,
            self.transformer_heads,
            self.attention_type,
        )
        self.fusion_norm = nn.LayerNorm(self.slot_dim) if self.fusion_refiner is not None else None
        self.decoder_stack = self._build_transformer_stack(
            self.decoder_transformer_layers,
            self.transformer_heads,
            self.attention_type,
        )
        self.decoder_norm = nn.LayerNorm(self.slot_dim) if self.decoder_stack is not None else None

    def rebuild_transformer_stacks(
        self,
        *,
        bank_layers: int,
        fixed_layers: int = 3,
        fusion_layers: int,
        decoder_layers: int,
        n_heads: int,
        attention_type: str,
        inherited_bank_layers: int | None = None,
        inherited_fusion_layers: int | None = None,
    ) -> None:
        inherited_bank = (
            int(inherited_bank_layers)
            if inherited_bank_layers is not None
            else int(getattr(self.transformer_policy, "inherited_bank_layers", bank_layers))
        )
        self.transformer_policy = DualTransformerPolicy.resolve(
            bank_layers_cfg=int(bank_layers),
            fixed_layers_cfg=int(fixed_layers),
            fusion_layers_cfg=int(fusion_layers),
            decoder_layers_cfg=int(decoder_layers),
            inherited_bank_layers=inherited_bank,
            inherited_fusion_layers=inherited_fusion_layers,
        )
        self._build_transformer_stacks(
            bank_layers=self.transformer_policy.bank_layers,
            fixed_layers=self.transformer_policy.fixed_layers,
            fusion_layers=self.transformer_policy.fusion_layers,
            decoder_layers=self.transformer_policy.decoder_layers,
            n_heads=n_heads,
            attention_type=attention_type,
        )

    def attach_wm_lattice_mirror(self, mirror: Any, store: Any = None) -> None:
        self.wm_lattice_mirror = mirror
        self.wm_shared_slot_store_mirror = store

    def _apply_transformer_stack(
        self,
        seq: torch.Tensor,
        stack: Optional[nn.Module],
        norm: Optional[nn.LayerNorm],
    ) -> torch.Tensor:
        if stack is None:
            return seq
        if isinstance(stack, nn.TransformerEncoder):
            return norm(seq + stack(seq)) if norm is not None else stack(seq)
        out = seq
        for block in stack:
            attn, block_norm = block[0], block[1]
            attn_out, _ = attn(out, out, out, need_weights=False)
            out = block_norm(out + attn_out)
        return out

    def _transform_sequence(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.dim() == 2:
            seq = vectors.unsqueeze(0)
            squeeze = True
        else:
            seq = vectors
            squeeze = False
        seq = self._apply_transformer_stack(seq, self.sequence_transformer, self.sequence_transformer_norm)
        seq = self._apply_transformer_stack(seq, self.aux_transformer, self.aux_norm)
        seq = self._apply_transformer_stack(seq, self.fusion_refiner, self.fusion_norm)
        return seq.squeeze(0) if squeeze else seq

    def _transform_episode_sequence(self, vectors: torch.Tensor) -> torch.Tensor:
        return self._transform_sequence(vectors)

    def _transform_query(self, query: torch.Tensor) -> torch.Tensor:
        return self._transform_sequence(query.unsqueeze(1)).squeeze(1)

    def _mirror_episode_to_wm_store(self, vectors: torch.Tensor, episode_id: str) -> None:
        store = self.wm_shared_slot_store_mirror
        if store is None or vectors.numel() == 0:
            return
        for i, vec in enumerate(vectors[:4]):
            if vec.numel() != store.config.dim:
                continue
            try:
                store.write_slot(
                    memory_type="hg_episodic",
                    local_slot_id=f"hg_episodic.{episode_id}.slot{i}",
                    content=vec.detach().reshape(-1),
                    owner="hg_ep_ltm",
                    geometry_map="hyperbolic",
                    depth_index=0,
                    write_permission=False,
                    metadata={"mirror_source": "hg_episodic_ltm", "episode_id": str(episode_id)},
                )
            except Exception:
                continue

    def qh_trace_summary(self) -> Dict[str, int]:
        return {
            "triplets_stored": int(self._qh_triplets_stored),
            "active_slots": int(len(self._qh_active_slots)),
        }

    def get_metrics(self) -> Dict[str, float]:
        return {
            "bank_transformer_layers": float(self.transformer_layers),
            "fixed_transformer_layers": float(self.fixed_transformer_layers),
            "fusion_transformer_layers": float(self.fusion_transformer_layers),
            "decoder_transformer_layers": float(self.decoder_transformer_layers),
            "dual_stack_active": float(self.fixed_transformer_layers > 0),
            "episode_records": float(len(self.episode_records)),
            "wm_lattice_mirror": float(self.wm_lattice_mirror is not None),
        }

    def set_external_attention_context(self, context: torch.Tensor) -> None:
        if context is None:
            self.external_attention_context = None
            return
        ctx = torch.as_tensor(
            context,
            device=self.slot_store.slot_values.device,
            dtype=self.slot_store.slot_values.dtype,
        )
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        elif ctx.dim() != 3:
            raise ValueError("external attention context must be [T,D], [B,D], or [B,S,D]")
        if ctx.size(-1) != self.slot_dim:
            raise ValueError(f"external attention context dim must be {self.slot_dim}")
        self.external_attention_context = ctx.detach()

    def clear_external_attention_context(self) -> None:
        self.external_attention_context = None

    @torch.no_grad()
    def build_episode_attention_context(
        self,
        *,
        episode_id: str,
        include_summary: bool = True,
        max_tokens: int = 128,
    ) -> Optional[torch.Tensor]:
        rec = self.episode_records.get(str(episode_id))
        if rec is None:
            return None
        # Memory-first context: keep compact summary anchors before raw slots.
        slot_ids: List[int] = []
        if include_summary and rec.summary_slot_ids:
            slot_ids.extend(list(rec.summary_slot_ids))
        slot_ids.extend(list(rec.slot_ids))
        if not slot_ids:
            return None
        values = self.slot_store.get_slot_value([int(s) for s in slot_ids])
        if values.numel() == 0:
            return None
        token_budget = max(1, int(max_tokens))
        if values.size(0) > token_budget and include_summary and rec.summary_slot_ids:
            # Keep summary tokens, then evenly sample the remainder from episode slots.
            summary_n = min(len(rec.summary_slot_ids), token_budget)
            summaries = values[:summary_n]
            remaining = token_budget - summary_n
            if remaining > 0:
                payload = values[summary_n:]
                if payload.size(0) > remaining:
                    idx = torch.linspace(0, payload.size(0) - 1, steps=remaining, device=payload.device)
                    payload = payload.index_select(0, idx.round().to(dtype=torch.long))
                values = torch.cat([summaries, payload], dim=0)
            else:
                values = summaries
        else:
            values = values[:token_budget]
        values = self._transform_episode_sequence(values)
        return values.unsqueeze(0)

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
            extra={**dict(extra), "qh_hologram": True},
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
        vectors = self._transform_episode_sequence(vectors)

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
        self._qh_triplets_stored += max(1, len(traces) * len(slot_ids))
        self._qh_active_slots.update(int(s) for s in slot_ids)

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
        self._mirror_episode_to_wm_store(vectors, episode_id)
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

    def _track_retrieval_use(self, slot_ids: torch.Tensor, metadata: Optional[List[List[Any]]] = None) -> None:
        if slot_ids.numel() == 0:
            return
        flat_ids = [int(x) for x in slot_ids.reshape(-1).tolist()]
        bonus_episode_slots: Set[int] = set()
        if metadata:
            episode_ids: Set[str] = set()
            for row in metadata:
                for meta in row:
                    extra = getattr(meta, "extra", {}) or {}
                    episode_id = extra.get("episode_id", None)
                    if episode_id:
                        episode_ids.add(str(episode_id))
            for episode_id in episode_ids:
                rec = self.episode_records.get(episode_id)
                if rec is None:
                    continue
                bonus_episode_slots.update(int(s) for s in (list(rec.slot_ids) + list(rec.summary_slot_ids)))
            if bonus_episode_slots:
                flat_set = set(flat_ids)
                for sid in sorted(bonus_episode_slots):
                    if sid not in flat_set:
                        flat_ids.append(int(sid))
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
        query_in = torch.as_tensor(
            query,
            device=self.slot_store.slot_values.device,
            dtype=self.slot_store.slot_values.dtype,
        )
        if query_in.dim() != 2 or query_in.size(1) != self.slot_dim:
            raise ValueError(f"query must be [B, D={self.slot_dim}]")
        if not torch.isfinite(query_in).all():
            raise ValueError("query contains NaN or Inf")
        query_in = self._transform_query(query_in)
        if self.external_attention_context is not None:
            ctx = self.external_attention_context.to(device=query_in.device, dtype=query_in.dtype)
            if ctx.size(0) == 1 and query_in.size(0) > 1:
                ctx = ctx.expand(query_in.size(0), -1, -1)
            elif ctx.size(0) != query_in.size(0):
                ctx = ctx.mean(dim=0, keepdim=True).expand(query_in.size(0), -1, -1)
            q = query_in.unsqueeze(1)
            qx, _ = self.query_context_attn(q, ctx, ctx, need_weights=False)
            query_in = self.query_context_norm(q + qx).squeeze(1)

        request = MemoryReadRequest(
            requester_system="hg_ep_ltm",
            query=query_in,
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

        if self.decoder_stack is not None and out.values.numel() > 0:
            refined = self._apply_transformer_stack(out.values, self.decoder_stack, self.decoder_norm)
            out = MemoryReadOutput(
                slot_ids=out.slot_ids,
                scores=out.scores,
                values=refined,
                metadata=out.metadata,
                diagnostics={**dict(out.diagnostics), "decoder_stack_applied": True},
            )

        self._track_retrieval_use(out.slot_ids, out.metadata)
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
