"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning regression closure.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid

from .reasoning_regression_matrix import build_reasoning_regression_matrix
from .reasoning_release_candidate import _safe_jsonable


class ReasoningRegressionClosureError(ValueError):
    """Raised when regression closure configuration is unsafe."""


@dataclass(frozen=True)
class ReasoningRegressionClosureConfig:
    """Regression closure config."""

    enabled: bool = False
    required_stages: tuple = (
        "REASON-1A",
        "REASON-1B",
        "REASON-1C",
        "REASON-1D",
        "REASON-1E",
        "REASON-2A",
        "REASON-2B",
        "REASON-2C",
        "REASON-2D",
        "REASON-3A",
        "REASON-3B",
        "REASON-3C",
        "REASON-3D",
    )
    require_safety_coverage: bool = True
    require_serialization_coverage: bool = True
    require_no_mutation_coverage: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if not self.required_stages:
            raise ReasoningRegressionClosureError("required_stages cannot be empty")
        if not self.no_mutation_by_default:
            raise ReasoningRegressionClosureError("no_mutation_by_default must remain true")

    @classmethod
    def disabled(cls) -> "ReasoningRegressionClosureConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_default(cls) -> "ReasoningRegressionClosureConfig":
        return cls(enabled=True)


@dataclass
class ReasoningRegressionClosureReport:
    """JSON-safe regression closure report."""

    enabled: bool
    closed: bool
    covered_stages: List[str] = field(default_factory=list)
    missing_stages: List[str] = field(default_factory=list)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)
    deferred_work: List[str] = field(default_factory=list)
    safety_flags: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: f"reasoning_regression_closure_{uuid.uuid4().hex[:16]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "report_id": self.report_id,
            "enabled": bool(self.enabled),
            "closed": bool(self.closed),
            "covered_stages": list(self.covered_stages),
            "missing_stages": list(self.missing_stages),
            "coverage_summary": _safe_jsonable(self.coverage_summary),
            "deferred_work": list(self.deferred_work),
            "safety_flags": _safe_jsonable(self.safety_flags),
            "lineage": _safe_jsonable(self.lineage),
        }
        json.dumps(payload, sort_keys=True)
        return payload


class ReasoningRegressionClosure:
    """Checks regression closure metadata without mutating code or stores."""

    def __init__(self, config: Optional[ReasoningRegressionClosureConfig] = None):
        self.config = config or ReasoningRegressionClosureConfig.disabled()
        self.config.validate()

    def close(self, *, lineage: Optional[Dict[str, Any]] = None) -> ReasoningRegressionClosureReport:
        if not self.config.enabled:
            return ReasoningRegressionClosureReport(
                enabled=False,
                closed=False,
                missing_stages=list(self.config.required_stages),
                safety_flags=self._safety_flags(),
                lineage=lineage or {},
            )

        matrix = build_reasoning_regression_matrix().to_dict()
        rows = matrix.get("rows", [])
        covered = sorted({str(row.get("stage")) for row in rows if isinstance(row, dict) and row.get("stage")})
        missing = [stage for stage in self.config.required_stages if stage not in covered]

        safety_rows = [row for row in rows if isinstance(row, dict) and row.get("safety_coverage")]
        serialization_rows = [row for row in rows if isinstance(row, dict) and row.get("serialization_coverage")]
        no_mutation_rows = [row for row in rows if isinstance(row, dict) and row.get("no_mutation_coverage")]

        blocking = []
        if missing:
            blocking.append("missing_stage_coverage")
        if self.config.require_safety_coverage and not safety_rows:
            blocking.append("missing_safety_coverage")
        if self.config.require_serialization_coverage and not serialization_rows:
            blocking.append("missing_serialization_coverage")
        if self.config.require_no_mutation_coverage and not no_mutation_rows:
            blocking.append("missing_no_mutation_coverage")

        report = ReasoningRegressionClosureReport(
            enabled=True,
            closed=not blocking,
            covered_stages=covered,
            missing_stages=missing,
            coverage_summary={
                "row_count": len(rows),
                "safety_rows": len(safety_rows),
                "serialization_rows": len(serialization_rows),
                "no_mutation_rows": len(no_mutation_rows),
                "blocking": blocking,
                "matrix_summary": matrix.get("summary", {}),
            },
            deferred_work=[
                "production persistence adapter tests",
                "external store integration tests",
                "long-run benchmark tests",
            ],
            safety_flags=self._safety_flags(),
            lineage=lineage or {},
        )
        json.dumps(report.to_dict(), sort_keys=True)
        return report

    @staticmethod
    def _safety_flags() -> Dict[str, Any]:
        return {
            "metadata_only": True,
            "permanent_memory_store_mutation": False,
            "model_weight_mutation": False,
            "optimizer_mutation": False,
            "destructive_replacement": False,
        }


def reasoning_regression_closure_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_regression_closure",
        "stage": "REASON-3D",
        "default_enabled": False,
        "metadata_only": True,
        "permanent_memory_store_mutation": False,
        "json_safe_report": True,
    }
