"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: confidence disagreement scoring.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math
import uuid

import torch

from .reasoning_orchestration_trace import _safe_jsonable


class ConfidenceDisagreementError(ValueError):
    """Raised when confidence/disagreement scoring receives unsafe inputs."""


@dataclass(frozen=True)
class ConfidenceDisagreementConfig:
    """Bounded, deterministic, non-mutating scoring config."""

    min_confidence: float = 0.05
    max_confidence: float = 0.98
    disagreement_weight: float = 0.55
    entropy_weight: float = 0.25
    support_weight: float = 0.20
    finite_checks: bool = True
    max_scores: int = 4096

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= self.max_confidence <= 1.0):
            raise ConfidenceDisagreementError("confidence bounds must satisfy 0 <= min <= max <= 1")
        total = self.disagreement_weight + self.entropy_weight + self.support_weight
        if total <= 0:
            raise ConfidenceDisagreementError("at least one scoring weight must be positive")
        if self.max_scores <= 0:
            raise ConfidenceDisagreementError("max_scores must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "disagreement_weight": self.disagreement_weight,
            "entropy_weight": self.entropy_weight,
            "support_weight": self.support_weight,
            "finite_checks": self.finite_checks,
            "max_scores": self.max_scores,
        }


@dataclass
class ConfidenceDisagreementReport:
    """JSON-safe confidence/disagreement report."""

    confidence: float
    disagreement: float
    support_mass: float
    entropy: float
    normalized_entropy: float
    method: str = "bounded_support_entropy"
    report_id: str = field(default_factory=lambda: f"conf_disagree_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "method": self.method,
            "confidence": float(self.confidence),
            "disagreement": float(self.disagreement),
            "support_mass": float(self.support_mass),
            "entropy": float(self.entropy),
            "normalized_entropy": float(self.normalized_entropy),
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "confidence_hook": True,
                "disagreement_hook": True,
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
            },
            "safety": {
                "bounded_scores": True,
                "non_mutating": True,
                "finite_checked": True,
            },
        }


def _validate_score_tensor(scores: torch.Tensor, config: ConfidenceDisagreementConfig) -> torch.Tensor:
    if not isinstance(scores, torch.Tensor):
        raise ConfidenceDisagreementError("scores must be a torch.Tensor")
    if scores.numel() == 0:
        raise ConfidenceDisagreementError("scores must not be empty")
    if scores.numel() > config.max_scores:
        raise ConfidenceDisagreementError(f"scores exceed max_scores={config.max_scores}")
    if config.finite_checks and not torch.isfinite(scores).all():
        raise ConfidenceDisagreementError("scores contain NaN/Inf")
    return scores.detach().clone().float().reshape(-1)


def score_confidence_disagreement(
    scores: torch.Tensor,
    *,
    support_mass: Optional[float] = None,
    config: Optional[ConfidenceDisagreementConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConfidenceDisagreementReport:
    """Compute bounded confidence/disagreement from finite support scores."""

    config = config or ConfidenceDisagreementConfig()
    config.validate()
    flat = _validate_score_tensor(scores, config)

    probabilities = torch.softmax(flat, dim=0)
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())
    max_entropy = math.log(max(int(probabilities.numel()), 2))
    normalized_entropy = float(max(0.0, min(1.0, entropy / max_entropy)))

    top = float(probabilities.max().item())
    second = float(torch.topk(probabilities, k=2).values[-1].item()) if probabilities.numel() > 1 else 0.0

    if support_mass is None:
        support_mass = float(flat.sigmoid().mean().item())
    support_mass = float(max(0.0, min(1.0, support_mass)))

    margin = max(0.0, min(1.0, top - second))
    raw_disagreement = (
        config.disagreement_weight * (1.0 - margin)
        + config.entropy_weight * normalized_entropy
        + config.support_weight * (1.0 - support_mass)
    )
    total_weight = config.disagreement_weight + config.entropy_weight + config.support_weight
    disagreement = float(max(0.0, min(1.0, raw_disagreement / total_weight)))
    confidence = float(max(config.min_confidence, min(config.max_confidence, 1.0 - disagreement)))

    return ConfidenceDisagreementReport(
        confidence=confidence,
        disagreement=disagreement,
        support_mass=support_mass,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        metadata={
            "score_count": int(flat.numel()),
            "top_probability": top,
            "second_probability": second,
            "margin": margin,
            **(metadata or {}),
        },
    )


def confidence_disagreement_contract() -> Dict[str, Any]:
    return {
        "module": "confidence_disagreement_scoring",
        "stage": "REASON-2B",
        "non_mutating": True,
        "finite_checks": True,
        "bounded_scores": True,
        "paamax_confidence_disagreement_hooks": True,
    }
