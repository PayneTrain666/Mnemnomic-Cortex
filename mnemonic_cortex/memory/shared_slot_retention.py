"""
Plain-language summary
----------------------
What this file is for: Shared-slot memory subsystem module: shared slot retention.
How it fits in the system: Manages shared memory slots that multiple systems can read/write under rules.
Status: OPT-IN
Important notes for non-coders: Not always enabled in standard capacity profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch

from .shared_slot_schema import code_to_slot_state
from .shared_slot_store import SharedSlotStore


@dataclass
class RetentionScore:
    slot_id: int
    keep_score: float
    demote_score: float
    evict_score: float
    rationale: Dict[str, float] = field(default_factory=dict)


class SharedSlotRetention:
    """
    Retention policy that scores each slot using:
    - usage
    - confidence
    - age
    - promotion_count
    - contradiction_count
    - state
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        usage_weight: float = 0.30,
        confidence_weight: float = 0.25,
        age_weight: float = 0.20,
        promotion_weight: float = 0.15,
        contradiction_penalty: float = 0.25,
    ) -> None:
        self.store = store
        self.usage_weight = float(usage_weight)
        self.confidence_weight = float(confidence_weight)
        self.age_weight = float(age_weight)
        self.promotion_weight = float(promotion_weight)
        self.contradiction_penalty = float(contradiction_penalty)

    def _state_keep_bias(self, state: str) -> float:
        # Positive means retain-biased, negative means evict-biased.
        table = {
            "free": -0.6,
            "volatile": -0.2,
            "provisional": 0.0,
            "durable": 0.45,
            "deprecated": -0.35,
            "quarantined": -0.55,
        }
        return float(table.get(state, 0.0))

    def _extract_counts(self, slot_id: int) -> tuple[float, float]:
        meta = self.store.metadata.get(int(slot_id))
        if meta is None:
            return 0.0, 0.0
        prov = getattr(meta, "provenance", None)
        if prov is None:
            return 0.0, 0.0
        promotions = float(getattr(prov, "promotion_count", 0.0) or 0.0)
        contradictions = float(getattr(prov, "contradiction_count", 0.0) or 0.0)
        return promotions, contradictions

    def score_slot(self, slot_id: int) -> RetentionScore:
        sid = int(slot_id)
        usage = float(self.store.slot_usage[sid].item())
        confidence = float(self.store.slot_confidence[sid].item())
        age = float(self.store.slot_age[sid].item())
        state = code_to_slot_state(int(self.store.slot_state_code[sid].item()))
        promotions, contradictions = self._extract_counts(sid)

        usage_norm = min(1.0, max(0.0, usage / 10.0))
        confidence_norm = min(1.0, max(0.0, confidence))
        # Older slots are slightly favored for retention if stable.
        age_norm = min(1.0, max(0.0, age / 1000.0))
        promotions_norm = min(1.0, max(0.0, promotions / 20.0))
        contradictions_norm = min(1.0, max(0.0, contradictions / 20.0))
        state_bias = self._state_keep_bias(state)

        keep_core = (
            (self.usage_weight * usage_norm)
            + (self.confidence_weight * confidence_norm)
            + (self.age_weight * age_norm)
            + (self.promotion_weight * promotions_norm)
            + state_bias
            - (self.contradiction_penalty * contradictions_norm)
        )
        # Map from unconstrained-ish score to [0,1].
        keep_score = float(max(0.0, min(1.0, 0.5 + (0.5 * keep_core))))
        evict_score = float(max(0.0, min(1.0, 1.0 - keep_score)))
        demote_score = float(max(0.0, min(1.0, evict_score * 0.75 + 0.25 * contradictions_norm)))

        return RetentionScore(
            slot_id=sid,
            keep_score=keep_score,
            demote_score=demote_score,
            evict_score=evict_score,
            rationale={
                "usage": usage,
                "confidence": confidence,
                "age": age,
                "promotion_count": promotions,
                "contradiction_count": contradictions,
                "state_bias": state_bias,
            },
        )

    def score_all_active_slots(self) -> List[RetentionScore]:
        out: List[RetentionScore] = []
        for slot_id in range(self.store.num_slots):
            state = code_to_slot_state(int(self.store.slot_state_code[slot_id].item()))
            if state == "free":
                continue
            out.append(self.score_slot(slot_id))
        return out

    def select_demotions(self, count: int) -> List[int]:
        if count <= 0:
            return []
        scores = self.score_all_active_slots()
        # Do not demote quarantined slots here; those should be reviewed explicitly.
        scores = [
            s
            for s in scores
            if code_to_slot_state(int(self.store.slot_state_code[s.slot_id].item())) != "quarantined"
        ]
        scores.sort(key=lambda s: s.demote_score, reverse=True)
        return [s.slot_id for s in scores[:count]]

    def select_evictions(self, count: int) -> List[int]:
        if count <= 0:
            return []
        scores = self.score_all_active_slots()
        # Keep durable slots unless no alternative exists.
        non_durable = [
            s
            for s in scores
            if code_to_slot_state(int(self.store.slot_state_code[s.slot_id].item())) != "durable"
        ]
        ranked = sorted(non_durable, key=lambda s: s.evict_score, reverse=True)
        if len(ranked) < count:
            durable_ranked = sorted(scores, key=lambda s: s.evict_score, reverse=True)
            ranked = durable_ranked
        return [s.slot_id for s in ranked[:count]]

    def apply_decay(self, delta_steps: int = 1) -> None:
        if delta_steps <= 0:
            return
        slot_ids = torch.arange(self.store.num_slots, device=self.store.slot_values.device, dtype=torch.long)
        self.store.increment_age(slot_ids, amount=int(delta_steps))
        decay = 0.995 ** float(delta_steps)
        self.store.slot_usage *= float(decay)
        self.store.version_counter += 1
