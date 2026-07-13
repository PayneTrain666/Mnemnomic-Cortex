"""
Plain-language summary
----------------------
What this file is for: Shared-slot memory subsystem module: shared slot arbitrator.
How it fits in the system: Manages shared memory slots that multiple systems can read/write under rules.
Status: OPT-IN
Important notes for non-coders: Not always enabled in standard capacity profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .shared_slot_schema import SlotWriteRequest, slot_state_to_code
from .shared_slot_store import SharedSlotStore


@dataclass
class SlotReadRequest:
    requester_system: str
    max_results: int = 8
    min_confidence: float = 0.0
    allow_quarantined: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitrationDecision:
    action: str  # allocate_new | overwrite | merge | reject | quarantine_candidate
    chosen_slot_ids: List[int]
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class SharedSlotArbitrator:
    def __init__(
        self,
        store: SharedSlotStore,
        *,
        overwrite_threshold: float = 0.35,
        merge_threshold: float = 0.65,
        quarantine_interference_threshold: float = 0.85,
    ) -> None:
        self.store = store
        self.overwrite_threshold = float(overwrite_threshold)
        self.merge_threshold = float(merge_threshold)
        self.quarantine_interference_threshold = float(quarantine_interference_threshold)

    def _slot_state_code(self, slot_id: int) -> int:
        return int(self.store.slot_state_code[int(slot_id)].item())

    def _is_slot_quarantined(self, slot_id: int) -> bool:
        return self._slot_state_code(slot_id) == slot_state_to_code("quarantined")

    def _is_slot_writable(self, slot_id: int, requester_system: str) -> bool:
        meta = self.store.metadata.get(int(slot_id))
        if meta is not None and hasattr(meta, "allowed_write_systems"):
            allowed = getattr(meta, "allowed_write_systems", []) or []
            return requester_system in allowed or len(allowed) == 0
        return True

    def _resolve_scores(
        self,
        candidate_slot_ids: Sequence[int],
        existing_scores: Optional[Sequence[float]],
    ) -> List[float]:
        if existing_scores is not None:
            scores = [float(x) for x in existing_scores]
            if len(scores) != len(candidate_slot_ids):
                raise ValueError("existing_scores length must match candidate_slot_ids length")
            return scores
        ids = torch.tensor(list(candidate_slot_ids), device=self.store.slot_values.device, dtype=torch.long)
        return self.store.slot_confidence.index_select(0, ids).detach().cpu().tolist()

    def score_merge_safety(self, *, slot_id: int, requester_system: str) -> float:
        conf = float(self.store.slot_confidence[int(slot_id)].item())
        usage = float(self.store.slot_usage[int(slot_id)].item())
        writable = 1.0 if self._is_slot_writable(slot_id, requester_system) else 0.0
        return (0.65 * conf) + (0.25 * min(1.0, usage / 10.0)) + (0.10 * writable)

    def score_merge_interference(self, *, slot_id: int, requester_system: str) -> float:
        conf = float(self.store.slot_confidence[int(slot_id)].item())
        usage = float(self.store.slot_usage[int(slot_id)].item())
        quarantined = 1.0 if self._is_slot_quarantined(slot_id) else 0.0
        return min(1.0, (0.55 * (1.0 - conf)) + (0.35 * min(1.0, usage / 10.0)) + (0.10 * quarantined))

    def score_merge_conflict(self, *, slot_id: int, requester_system: str) -> float:
        meta = self.store.metadata.get(int(slot_id))
        if meta is None:
            return 0.0
        primary = getattr(meta, "primary_system_id", None)
        state = getattr(meta, "state", None)
        conflict = 0.0
        if primary and primary != requester_system:
            conflict += 0.4
        if state in ("durable", "quarantined"):
            conflict += 0.6
        return min(1.0, conflict)

    def decide_merge(
        self,
        *,
        requester_system: str,
        candidate_slot_ids: Sequence[int],
    ) -> ArbitrationDecision:
        if not candidate_slot_ids:
            return ArbitrationDecision(action="reject", chosen_slot_ids=[], reason="no_candidate_slots")

        scored: List[tuple[float, int, Dict[str, float]]] = []
        for slot_id in candidate_slot_ids:
            safety = self.score_merge_safety(slot_id=int(slot_id), requester_system=requester_system)
            interference = self.score_merge_interference(slot_id=int(slot_id), requester_system=requester_system)
            conflict = self.score_merge_conflict(slot_id=int(slot_id), requester_system=requester_system)
            aggregate = safety - (0.55 * interference) - (0.45 * conflict)
            scored.append(
                (
                    aggregate,
                    int(slot_id),
                    {"safety": safety, "interference": interference, "conflict": conflict},
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_slot, best_diag = scored[0]
        if best_diag["interference"] >= self.quarantine_interference_threshold:
            return ArbitrationDecision(
                action="quarantine_candidate",
                chosen_slot_ids=[best_slot],
                reason="high_interference_detected",
                diagnostics={"best_score": best_score, **best_diag},
            )
        if best_score >= self.merge_threshold:
            return ArbitrationDecision(
                action="merge",
                chosen_slot_ids=[best_slot],
                reason="merge_threshold_met",
                diagnostics={"best_score": best_score, **best_diag},
            )
        return ArbitrationDecision(
            action="allocate_new",
            chosen_slot_ids=[],
            reason="merge_threshold_not_met",
            diagnostics={"best_score": best_score, **best_diag},
        )

    def decide_write(
        self,
        *,
        request: SlotWriteRequest,
        candidate_slot_ids: Sequence[int],
        existing_scores: Optional[Sequence[float]] = None,
    ) -> ArbitrationDecision:
        candidates = [int(x) for x in candidate_slot_ids]
        if not candidates:
            return ArbitrationDecision(
                action="allocate_new",
                chosen_slot_ids=[],
                reason="no_candidates",
                diagnostics={"requester_system": request.requester_system},
            )

        scores = self._resolve_scores(candidates, existing_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        best_slot, best_score = ranked[0]
        writable = self._is_slot_writable(best_slot, request.requester_system)
        quarantined = self._is_slot_quarantined(best_slot)

        if quarantined:
            return ArbitrationDecision(
                action="quarantine_candidate",
                chosen_slot_ids=[best_slot],
                reason="candidate_is_quarantined",
                diagnostics={"best_score": best_score},
            )
        if not writable:
            return ArbitrationDecision(
                action="reject",
                chosen_slot_ids=[],
                reason="write_not_allowed_for_requester",
                diagnostics={"best_slot": best_slot, "best_score": best_score},
            )

        if best_score <= self.overwrite_threshold:
            return ArbitrationDecision(
                action="overwrite",
                chosen_slot_ids=[best_slot],
                reason="low_confidence_slot_reuse",
                diagnostics={"best_score": best_score},
            )

        merge_decision = self.decide_merge(
            requester_system=request.requester_system,
            candidate_slot_ids=[slot for slot, _ in ranked[: max(1, min(4, len(ranked)))]],
        )
        if merge_decision.action == "merge":
            return merge_decision

        return ArbitrationDecision(
            action="allocate_new",
            chosen_slot_ids=[],
            reason="high_confidence_existing_slot",
            diagnostics={"best_slot": best_slot, "best_score": best_score},
        )

    def decide_read(
        self,
        *,
        request: SlotReadRequest,
        candidate_slot_ids: Sequence[int],
        existing_scores: Optional[Sequence[float]] = None,
    ) -> ArbitrationDecision:
        candidates = [int(x) for x in candidate_slot_ids]
        if not candidates:
            return ArbitrationDecision(
                action="reject",
                chosen_slot_ids=[],
                reason="no_candidates",
                diagnostics={"requester_system": request.requester_system},
            )

        scores = self._resolve_scores(candidates, existing_scores)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        selected: List[int] = []
        for slot_id, score in ranked:
            if score < request.min_confidence:
                continue
            if not request.allow_quarantined and self._is_slot_quarantined(slot_id):
                continue
            selected.append(slot_id)
            if len(selected) >= int(request.max_results):
                break

        if not selected:
            return ArbitrationDecision(
                action="reject",
                chosen_slot_ids=[],
                reason="no_readable_slots_after_filters",
                diagnostics={
                    "min_confidence": request.min_confidence,
                    "max_results": request.max_results,
                    "candidate_count": len(candidates),
                },
            )

        return ArbitrationDecision(
            action="merge" if len(selected) > 1 else "overwrite",
            chosen_slot_ids=selected,
            reason="read_candidates_selected",
            diagnostics={
                "min_confidence": request.min_confidence,
                "max_results": request.max_results,
                "selected_count": len(selected),
            },
        )
