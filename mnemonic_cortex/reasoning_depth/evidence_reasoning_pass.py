from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import uuid

import torch

from .reasoning_orchestration_trace import _safe_jsonable


class EvidenceReasoningError(ValueError):
    """Raised when evidence reasoning receives unsafe input."""


@dataclass(frozen=True)
class EvidenceReasoningConfig:
    """Bounded evidence-structured reasoning config.

    Evidence extraction is deliberately simple in REASON-2C. It creates
    serialization-safe units from provided content and optional support tensors.
    It does not mutate memory stores, model weights, optimizer state, or source
    tensors.
    """

    enabled: bool = False
    max_evidence_units: int = 12
    max_text_chars: int = 4096
    min_unit_chars: int = 1
    finite_checks: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.max_evidence_units <= 0 or self.max_evidence_units > 128:
            raise EvidenceReasoningError("max_evidence_units must be in [1,128]")
        if self.max_text_chars <= 0:
            raise EvidenceReasoningError("max_text_chars must be positive")
        if self.min_unit_chars <= 0:
            raise EvidenceReasoningError("min_unit_chars must be positive")

    @classmethod
    def disabled(cls) -> "EvidenceReasoningConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "EvidenceReasoningConfig":
        return cls(enabled=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_evidence_units": self.max_evidence_units,
            "max_text_chars": self.max_text_chars,
            "min_unit_chars": self.min_unit_chars,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
        }


@dataclass
class EvidenceUnit:
    """A compact, JSON-safe evidence unit."""

    text: str
    source: str = "content"
    support: float = 0.5
    unit_id: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id:
            digest = hashlib.sha256(f"{self.source}|{self.text}".encode("utf-8")).hexdigest()[:16]
            self.unit_id = f"evidence.{digest}"
        self.support = float(max(0.0, min(1.0, self.support)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "source": self.source,
            "support": self.support,
            "tags": list(self.tags),
            "metadata": _safe_jsonable(self.metadata),
        }


@dataclass
class EvidenceReasoningReport:
    """Report emitted by an evidence reasoning pass."""

    evidence_units: List[EvidenceUnit]
    enabled: bool
    report_id: str = field(default_factory=lambda: f"evidence_report_{uuid.uuid4().hex[:16]}")
    aggregate_support: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_units:
            self.aggregate_support = float(sum(unit.support for unit in self.evidence_units) / len(self.evidence_units))
        else:
            self.aggregate_support = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "enabled": self.enabled,
            "evidence_units": [unit.to_dict() for unit in self.evidence_units],
            "aggregate_support": float(self.aggregate_support),
            "metadata": _safe_jsonable(self.metadata),
            "paamax_metadata": {
                "trace_governance": True,
                "evidence_units": True,
                "audit_metadata": True,
                "write_permission_required_for_commit": True,
            },
            "safety": {
                "bounded_units": True,
                "non_mutating": True,
                "finite_checked": True,
            },
        }


class EvidenceReasoningPass:
    """Bounded evidence-structured reasoning pass."""

    def __init__(self, config: Optional[EvidenceReasoningConfig] = None):
        self.config = config or EvidenceReasoningConfig.disabled()
        self.config.validate()

    def run(
        self,
        *,
        content: str = "",
        query: Optional[torch.Tensor] = None,
        support_scores: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReasoningReport:
        if query is not None:
            self._validate_tensor(query, "query")
        if support_scores is not None:
            self._validate_tensor(support_scores, "support_scores")

        if not self.config.enabled:
            return EvidenceReasoningReport(
                evidence_units=[],
                enabled=False,
                metadata={"reason": "evidence reasoning disabled", **(metadata or {})},
            )

        text = (content or "")[: self.config.max_text_chars]
        fragments = self._split_content(text)
        support_values = self._support_values(len(fragments), support_scores)
        units = [
            EvidenceUnit(
                text=frag,
                source="content",
                support=support_values[i],
                tags=["reason2c", "evidence"],
                metadata={"index": i, "content_chars": len(frag)},
            )
            for i, frag in enumerate(fragments)
        ]
        return EvidenceReasoningReport(
            evidence_units=units,
            enabled=True,
            metadata={"content_truncated": len(content or "") > self.config.max_text_chars, **(metadata or {})},
        )

    def _split_content(self, text: str) -> List[str]:
        if not text:
            return []
        raw_parts = []
        for sentence in text.replace("\n", " ").split("."):
            sentence = sentence.strip()
            if len(sentence) >= self.config.min_unit_chars:
                raw_parts.append(sentence)
        if not raw_parts and text.strip():
            raw_parts = [text.strip()]
        return raw_parts[: self.config.max_evidence_units]

    def _support_values(self, count: int, support_scores: Optional[torch.Tensor]) -> List[float]:
        if count <= 0:
            return []
        if support_scores is None:
            return [1.0 / count for _ in range(count)]
        flat = support_scores.detach().clone().float().reshape(-1)
        if flat.numel() == 0:
            return [1.0 / count for _ in range(count)]
        probs = torch.softmax(flat[: max(count, 1)], dim=0)
        values = [float(probs[min(i, probs.numel() - 1)].item()) for i in range(count)]
        return [float(max(0.0, min(1.0, value))) for value in values]

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise EvidenceReasoningError(f"{name} must be a torch.Tensor")
        if self.config.finite_checks and not torch.isfinite(tensor).all():
            raise EvidenceReasoningError(f"{name} contains NaN/Inf")


def evidence_reasoning_contract() -> Dict[str, Any]:
    return {
        "module": "evidence_reasoning_pass",
        "stage": "REASON-2C",
        "default_enabled": False,
        "bounded_evidence_units": True,
        "non_mutating": True,
        "paamax_evidence_metadata": True,
    }
