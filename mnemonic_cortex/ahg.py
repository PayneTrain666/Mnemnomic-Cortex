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
        for name, value in {
            "proto_tau": self.proto_tau,
            "fisher_tau": self.fisher_tau,
            "phase_rho": self.phase_rho,
            "agree_tau": self.agree_tau,
        }.items():
            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if float(self.proto_tau) < 0.0:
            raise ValueError("proto_tau must be >= 0")
        if float(self.fisher_tau) < 0.0:
            raise ValueError("fisher_tau must be >= 0")
        if not (0.0 <= float(self.phase_rho) <= 1.0):
            raise ValueError("phase_rho must be in [0,1]")
        if not (0.0 <= float(self.agree_tau) <= 1.0):
            raise ValueError("agree_tau must be in [0,1]")


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

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            out = float(value)
            if math.isfinite(out):
                return out
        except Exception:
            pass
        return float(default)

    def decide(self, broker_result: Dict[str, Any], cross_diag: Optional[Dict[str, Any]] = None) -> AHGDecision:
        signals = broker_result.get("signals", {}) or {}
        proto = self._safe_float(signals.get("proto_distance", 1e9), 1e9)
        phase = self._safe_float(signals.get("phase_agreement", 0.0), 0.0)
        fisher = self._safe_float(signals.get("fisher_uncertainty", 1e9), 1e9)
        has_fisher = bool(broker_result.get("signals", {}).get("has_fisher", False))
        agree = self._safe_float((cross_diag or {}).get("agreement", 0.0), 0.0)

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

