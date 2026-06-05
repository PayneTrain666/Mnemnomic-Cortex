from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .shared_slot_retention import SharedSlotRetention
from .shared_slot_schema import SlotMetadata, code_to_slot_state, slot_state_to_code
from .shared_slot_store import SharedSlotStore


@dataclass
class LifecycleDecision:
    slot_id: int
    old_state: str
    new_state: str
    reason: str
    confidence: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class MemoryLifecycleManager:
    """
    Lifecycle doctrine for shared-slot state transitions.

    Promotion doctrine:
      volatile/provisional -> durable when usage/confidence are strong, contradiction is low,
      provenance is acceptable, and truth maintenance does not object.

    Demotion doctrine:
      durable/provisional -> lower confidence tier when contradiction rises, usage collapses,
      confidence decays, or downstream truth checks fail.
    """

    def __init__(
        self,
        *,
        store: SharedSlotStore,
        retention: SharedSlotRetention,
        truth_runtime: Any = None,
        sustained_usage_threshold: float = 2.5,
        high_confidence_threshold: float = 0.75,
        low_confidence_threshold: float = 0.35,
        low_contradiction_threshold: int = 1,
        high_contradiction_threshold: int = 4,
    ) -> None:
        self.store = store
        self.retention = retention
        self.truth_runtime = truth_runtime
        self.sustained_usage_threshold = float(sustained_usage_threshold)
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.low_contradiction_threshold = int(low_contradiction_threshold)
        self.high_contradiction_threshold = int(high_contradiction_threshold)

    def _slot_state(self, slot_id: int) -> str:
        return code_to_slot_state(int(self.store.slot_state_code[int(slot_id)].item()))

    def _slot_metadata(self, slot_id: int) -> SlotMetadata | None:
        meta = self.store.metadata.get(int(slot_id))
        return meta if isinstance(meta, SlotMetadata) else None

    def _contradiction_count(self, slot_id: int) -> int:
        meta = self._slot_metadata(slot_id)
        prov = meta.provenance if meta is not None else None
        return int(getattr(prov, "contradiction_count", 0) or 0)

    def _provenance_acceptable(self, slot_id: int) -> bool:
        meta = self._slot_metadata(slot_id)
        if meta is None or meta.provenance is None:
            return False
        src = str(meta.provenance.source_system or "").strip()
        traces = list(meta.provenance.source_trace_ids or [])
        return bool(src) and len(traces) > 0

    def _truth_allows_promotion(self, slot_id: int, diagnostics: Dict[str, Any]) -> bool:
        runtime = self.truth_runtime
        if runtime is None:
            return True
        fn = getattr(runtime, "allows_promotion", None)
        if callable(fn):
            return bool(fn(slot_id=int(slot_id), diagnostics=dict(diagnostics)))
        return True

    def _downstream_checks_pass(self, slot_id: int, diagnostics: Dict[str, Any]) -> bool:
        runtime = self.truth_runtime
        if runtime is None:
            return True
        fn = getattr(runtime, "downstream_checks_pass", None)
        if callable(fn):
            return bool(fn(slot_id=int(slot_id), diagnostics=dict(diagnostics)))
        return True

    def evaluate_promotion(self, slot_id: int) -> LifecycleDecision:
        sid = int(slot_id)
        old_state = self._slot_state(sid)
        usage = float(self.store.slot_usage[sid].item())
        confidence = float(self.store.slot_confidence[sid].item())
        contradictions = self._contradiction_count(sid)
        retention_score = self.retention.score_slot(sid)
        provenance_ok = self._provenance_acceptable(sid)
        truth_ok = self._truth_allows_promotion(
            sid,
            diagnostics={
                "usage": usage,
                "confidence": confidence,
                "contradiction_count": contradictions,
                "keep_score": retention_score.keep_score,
            },
        )

        checks = {
            "state_eligible": old_state in {"volatile", "provisional"},
            "sustained_usage": usage >= self.sustained_usage_threshold,
            "high_confidence": confidence >= self.high_confidence_threshold,
            "low_contradiction": contradictions <= self.low_contradiction_threshold,
            "provenance_acceptable": provenance_ok,
            "truth_maintenance_clear": truth_ok,
        }
        promote = all(bool(v) for v in checks.values())
        new_state = "durable" if promote else old_state
        reason = "promote_to_durable" if promote else "promotion_thresholds_not_met"
        return LifecycleDecision(
            slot_id=sid,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            confidence=float(retention_score.keep_score),
            diagnostics={
                **checks,
                "usage": usage,
                "confidence_score": confidence,
                "contradiction_count": contradictions,
                "retention_keep_score": retention_score.keep_score,
            },
        )

    def evaluate_demotion(self, slot_id: int) -> LifecycleDecision:
        sid = int(slot_id)
        old_state = self._slot_state(sid)
        usage = float(self.store.slot_usage[sid].item())
        confidence = float(self.store.slot_confidence[sid].item())
        contradictions = self._contradiction_count(sid)
        retention_score = self.retention.score_slot(sid)
        downstream_ok = self._downstream_checks_pass(
            sid,
            diagnostics={
                "usage": usage,
                "confidence": confidence,
                "contradiction_count": contradictions,
                "demote_score": retention_score.demote_score,
            },
        )

        checks = {
            "state_eligible": old_state in {"durable", "provisional"},
            "contradiction_burden_rises": contradictions >= self.high_contradiction_threshold,
            "usage_collapses": usage < self.sustained_usage_threshold * 0.40,
            "confidence_decays": confidence < self.low_confidence_threshold,
            "downstream_checks_fail": not downstream_ok,
        }
        trigger = (
            checks["state_eligible"]
            and (
                checks["contradiction_burden_rises"]
                or checks["usage_collapses"]
                or checks["confidence_decays"]
                or checks["downstream_checks_fail"]
            )
        )
        if trigger and old_state == "durable":
            new_state = "provisional"
        elif trigger and old_state == "provisional":
            new_state = "volatile"
        else:
            new_state = old_state
        reason = "demote_state" if trigger else "demotion_thresholds_not_met"
        return LifecycleDecision(
            slot_id=sid,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            confidence=float(retention_score.demote_score),
            diagnostics={
                **checks,
                "usage": usage,
                "confidence_score": confidence,
                "contradiction_count": contradictions,
                "retention_demote_score": retention_score.demote_score,
            },
        )

    def evaluate_quarantine(self, slot_id: int) -> LifecycleDecision:
        sid = int(slot_id)
        old_state = self._slot_state(sid)
        contradictions = self._contradiction_count(sid)
        retention_score = self.retention.score_slot(sid)
        severe = contradictions >= max(self.high_contradiction_threshold + 3, 7)
        downstream_fail = not self._downstream_checks_pass(
            sid,
            diagnostics={"contradiction_count": contradictions, "evict_score": retention_score.evict_score},
        )
        quarantine = old_state != "free" and severe and downstream_fail
        return LifecycleDecision(
            slot_id=sid,
            old_state=old_state,
            new_state="quarantined" if quarantine else old_state,
            reason="quarantine_slot" if quarantine else "quarantine_not_required",
            confidence=float(retention_score.evict_score),
            diagnostics={
                "severe_contradictions": severe,
                "downstream_checks_fail": downstream_fail,
                "contradiction_count": contradictions,
                "retention_evict_score": retention_score.evict_score,
            },
        )

    def apply_decision(self, decision: LifecycleDecision) -> None:
        sid = int(decision.slot_id)
        if decision.new_state == decision.old_state:
            return
        self.store.set_slot_state_code(
            slot_ids=[sid],
            state_code=[slot_state_to_code(decision.new_state)],
        )
        meta = self._slot_metadata(sid)
        if meta is None:
            return
        meta.state = decision.new_state  # keep metadata aligned with buffer state
        if meta.provenance is not None:
            if decision.old_state in {"volatile", "provisional"} and decision.new_state == "durable":
                meta.provenance.promotion_count = int(meta.provenance.promotion_count) + 1
            if decision.new_state in {"provisional", "volatile", "quarantined"}:
                meta.provenance.contradiction_count = max(
                    int(meta.provenance.contradiction_count),
                    int(self._contradiction_count(sid)),
                )
            meta.provenance.last_update_step = int(self.store.version_counter)
        extra = dict(meta.extra or {})
        extra["lifecycle_last_reason"] = decision.reason
        extra["lifecycle_last_confidence"] = float(decision.confidence)
        meta.extra = extra
        self.store.set_slot_metadata(slot_ids=[sid], metadata={sid: meta})

    def evaluate_cycle(self, slot_ids: List[int]) -> List[LifecycleDecision]:
        decisions: List[LifecycleDecision] = []
        for sid in slot_ids:
            quarantine = self.evaluate_quarantine(int(sid))
            if quarantine.new_state != quarantine.old_state:
                decisions.append(quarantine)
                continue
            promote = self.evaluate_promotion(int(sid))
            if promote.new_state != promote.old_state:
                decisions.append(promote)
                continue
            demote = self.evaluate_demotion(int(sid))
            decisions.append(demote)
        return decisions
