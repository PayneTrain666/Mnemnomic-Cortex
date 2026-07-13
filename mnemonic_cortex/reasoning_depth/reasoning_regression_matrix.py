"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning regression matrix.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json


@dataclass
class RegressionMatrixRow:
    stage: str
    module: str
    test_file: str
    safety_coverage: bool
    serialization_coverage: bool
    no_mutation_coverage: bool
    deferred_work: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "module": self.module,
            "test_file": self.test_file,
            "safety_coverage": bool(self.safety_coverage),
            "serialization_coverage": bool(self.serialization_coverage),
            "no_mutation_coverage": bool(self.no_mutation_coverage),
            "deferred_work": self.deferred_work,
        }


@dataclass
class ReasoningRegressionMatrix:
    rows: List[RegressionMatrixRow] = field(default_factory=list)
    stage: str = "REASON-3D"

    def to_dict(self) -> Dict[str, Any]:
        row_payloads = [row.to_dict() for row in self.rows]
        summary = {
            "stage": self.stage,
            "total_rows": len(row_payloads),
            "safety_coverage_rows": sum(1 for row in row_payloads if row.get("safety_coverage")),
            "serialization_coverage_rows": sum(1 for row in row_payloads if row.get("serialization_coverage")),
            "no_mutation_coverage_rows": sum(1 for row in row_payloads if row.get("no_mutation_coverage")),
            "json_safe": True,
        }
        payload = {
            "stage": self.stage,
            "rows": row_payloads,
            "summary": summary,
            "row_count": len(row_payloads),
            "json_safe": True,
        }
        json.dumps(payload, sort_keys=True)
        return payload


def build_reasoning_regression_matrix() -> ReasoningRegressionMatrix:
    rows = [
        RegressionMatrixRow("REASON-1A", "depth_lattice", "tests/test_reason1a_*", True, True, True, "WM/MANN/LTM integration staged"),
        RegressionMatrixRow("REASON-1B", "wm_depth_adapter", "tests/test_reason1b_*", True, True, True, "MANN integration staged"),
        RegressionMatrixRow("REASON-1C", "mann_depth_adapter", "tests/test_reason1c_*", True, True, True, "LTM integration staged"),
        RegressionMatrixRow("REASON-1D", "ltm_depth_adapter", "tests/test_reason1d_*", True, True, True, "benchmarks staged"),
        RegressionMatrixRow("REASON-1E", "capacity_validation", "tests/test_reason1e_*", True, True, True, "controller staged"),
        RegressionMatrixRow("REASON-2A", "reasoning_controller", "tests/test_reason2a_*", True, True, True, "policy router staged"),
        RegressionMatrixRow("REASON-2B", "policy_router", "tests/test_reason2b_*", True, True, True, "evidence/counterfactual staged"),
        RegressionMatrixRow("REASON-2C", "evidence_counterfactual_conflict", "tests/test_reason2c_*", True, True, True, "API hardening staged"),
        RegressionMatrixRow("REASON-2D", "controller_api_release_audit", "tests/test_reason2d_*", True, True, True, "strategy graph staged"),
        RegressionMatrixRow("REASON-3A", "strategy_graph_planner_route_expander", "tests/test_reason3a_*", True, True, True, "planner evaluation staged"),
        RegressionMatrixRow("REASON-3B", "planner_evaluation_failure_remediation", "tests/test_reason3b_*", True, True, True, "planner quality staged"),
        RegressionMatrixRow("REASON-3C", "planner_quality_controller_integration_persistence_readiness", "tests/test_reason3c_*", True, True, True, "release candidate staged"),
        RegressionMatrixRow("REASON-3D", "release_candidate_api_freeze_regression_closure", "tests/test_reason3d_*", True, True, True, "optional persistence adapter design"),
    ]
    return ReasoningRegressionMatrix(rows=rows)


def reasoning_regression_matrix_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_regression_matrix",
        "stage": "REASON-3D",
        "json_safe_export": True,
        "safety_coverage": True,
        "serialization_coverage": True,
        "no_mutation_coverage": True,
    }
