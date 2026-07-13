"""
Plain-language summary
----------------------
What this file is for: Anti-Hallucination Guard: decides when an answer looks unsupported.
How it fits in the system: Safety layer that can block or flag risky outputs.
Status: OPT-IN
Important notes for non-coders: Enable when evaluating truthfulness-sensitive tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math


@dataclass
class AHGConfig:
    proto_tau: float = 0.45
    fisher_tau: float = 0.60
    phase_rho: float = 0.10
    agree_tau: float = 0.15
    ask_on_uncertain: bool = True
    refuse_on_high_risk: bool = True

    def validate(self) -> None:
        for name in ("proto_tau", "fisher_tau", "phase_rho", "agree_tau"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class AHGDecision:
    action: str
    reason: str
    scores: Dict[str, float]


class AntiHallucinationGuard:
    """
    Broker-signal based risk gate for recall policy selection.
    """

    def __init__(self, cfg: AHGConfig):
        cfg.validate()
        self.cfg = cfg

    def decide(self, broker_result: Dict[str, Any], cross_diag: Optional[Dict[str, Any]] = None) -> AHGDecision:
        def _safe(value: Any, fallback: float) -> float:
            x = float(value)
            return x if math.isfinite(x) else fallback

        proto = _safe(broker_result.get("signals", {}).get("proto_distance", 1e9), 1e9)
        phase = _safe(broker_result.get("signals", {}).get("phase_agreement", 0.0), 0.0)
        fisher = _safe(broker_result.get("signals", {}).get("fisher_uncertainty", 1e9), 1e9)
        has_fisher = bool(broker_result.get("signals", {}).get("has_fisher", False))
        agree = _safe((cross_diag or {}).get("agreement", 0.0), 0.0)

        proto_ok = proto <= self.cfg.proto_tau
        phase_ok = phase >= self.cfg.phase_rho
        fisher_ok = (not has_fisher) or (fisher <= self.cfg.fisher_tau)
        agree_ok = agree >= self.cfg.agree_tau

        scores = {"proto": proto, "phase": phase, "fisher": fisher, "agree": agree}
        if proto_ok and phase_ok and fisher_ok and agree_ok:
            return AHGDecision("explosive", "Strong evidence and agreement.", scores)
        if proto_ok and fisher_ok:
            return AHGDecision("allow", "Prototype match with acceptable uncertainty.", scores)
        if (not proto_ok) and fisher_ok and self.cfg.ask_on_uncertain:
            return AHGDecision("refine", "Need refinement/verification before answer.", scores)
        if self.cfg.refuse_on_high_risk:
            return AHGDecision("ask", "High risk profile; request more context/sources.", scores)
        return AHGDecision("allow", "Policy allows continuation despite weak signals.", scores)

