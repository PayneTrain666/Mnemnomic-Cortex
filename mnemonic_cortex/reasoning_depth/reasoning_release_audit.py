from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import importlib
import json
import uuid

import torch

from .reasoning_controller_api import ReasoningControllerAPI, ReasoningControllerAPIConfig
from .reasoning_orchestration_trace import _safe_jsonable


class ReasoningReleaseAuditError(ValueError):
    """Raised when release audit inputs are unsafe."""


@dataclass(frozen=True)
class ReasoningReleaseAuditConfig:
    """Configuration for bounded release-readiness audit."""

    enabled: bool = True
    source_pack: str = ""
    stage: str = "REASON-2D"
    max_module_checks: int = 64
    require_disabled_default: bool = True
    require_no_mutation: bool = True
    require_json_serialization: bool = True

    def validate(self) -> None:
        if self.max_module_checks <= 0 or self.max_module_checks > 512:
            raise ReasoningReleaseAuditError("max_module_checks must be in [1,512]")


@dataclass
class ReasoningReleaseAuditReport:
    """JSON-safe release audit report."""

    stage: str
    source_pack: str
    module_availability: Dict[str, bool]
    public_import_availability: Dict[str, bool]
    disabled_default_verified: bool
    no_mutation_verified: bool
    config_serialization_verified: bool
    trace_serialization_verified: bool
    optional_passes_verified: bool
    remaining_deferred_work: List[str]
    readiness_level: str
    report_id: str = field(default_factory=lambda: f"release_audit_{uuid.uuid4().hex[:16]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "stage": self.stage,
            "source_pack": self.source_pack,
            "module_availability": dict(self.module_availability),
            "public_import_availability": dict(self.public_import_availability),
            "disabled_default_verified": self.disabled_default_verified,
            "no_mutation_verified": self.no_mutation_verified,
            "config_serialization_verified": self.config_serialization_verified,
            "trace_serialization_verified": self.trace_serialization_verified,
            "optional_passes_verified": self.optional_passes_verified,
            "remaining_deferred_work": list(self.remaining_deferred_work),
            "readiness_level": self.readiness_level,
            "metadata": _safe_jsonable(self.metadata),
            "safety": {
                "non_mutating_audit": True,
                "no_fake_production_complete_claim": True,
                "qh_metadata_only": True,
            },
        }


def run_reasoning_release_audit(config: Optional[ReasoningReleaseAuditConfig] = None) -> ReasoningReleaseAuditReport:
    cfg = config or ReasoningReleaseAuditConfig()
    cfg.validate()

    modules = [
        "mnemonic_cortex.reasoning_depth.reasoning_controller",
        "mnemonic_cortex.reasoning_depth.reasoning_controller_api",
        "mnemonic_cortex.reasoning_depth.evidence_reasoning_pass",
        "mnemonic_cortex.reasoning_depth.counterfactual_reasoning_probe",
        "mnemonic_cortex.reasoning_depth.conflict_aware_consolidation",
        "mnemonic_cortex.reasoning_depth.reasoning_policy_router",
        "mnemonic_cortex.reasoning_depth.consolidation_gate",
    ][: cfg.max_module_checks]

    module_availability: Dict[str, bool] = {}
    for name in modules:
        try:
            importlib.import_module(name)
            module_availability[name] = True
        except Exception:
            module_availability[name] = False

    public_names = [
        "ReasoningControllerAPI",
        "ReasoningControllerAPIConfig",
        "ReasoningController",
        "ReasoningControllerConfig",
        "EvidenceReasoningPass",
        "CounterfactualReasoningProbe",
        "ConflictAwareConsolidationEvaluator",
    ]
    public_import_availability: Dict[str, bool] = {}
    try:
        root = importlib.import_module("mnemonic_cortex.reasoning_depth")
        for name in public_names:
            public_import_availability[name] = hasattr(root, name)
    except Exception:
        for name in public_names:
            public_import_availability[name] = False

    disabled_api = ReasoningControllerAPI(ReasoningControllerAPIConfig.disabled(key_dim=8))
    disabled_default_verified = not disabled_api.config.enabled

    x = torch.randn(1, 2, 8)
    before = x.clone()
    result = disabled_api.run_reasoning_pass(x, content="audit")
    no_mutation_verified = torch.equal(x, before) and result.to_dict()["reasoning_result"]["mutation_performed"] is False

    config_serialization_verified = False
    trace_serialization_verified = False
    try:
        payload = disabled_api.config.to_dict()
        rebuilt = ReasoningControllerAPIConfig.from_dict(payload)
        json.dumps(rebuilt.to_dict(), sort_keys=True)
        config_serialization_verified = True
        json.dumps(result.to_dict(), sort_keys=True)
        trace_serialization_verified = True
    except Exception:
        config_serialization_verified = False
        trace_serialization_verified = False

    optional_passes_verified = all(module_availability.get(name, False) for name in modules[2:5])
    all_core_ready = (
        all(module_availability.values())
        and all(public_import_availability.values())
        and disabled_default_verified
        and no_mutation_verified
        and config_serialization_verified
        and trace_serialization_verified
        and optional_passes_verified
    )
    readiness_level = "release_candidate" if all_core_ready else "needs_patch"
    remaining_deferred_work = [
        "REASON-3A advanced reasoning strategy graph and multi-pass thought planner",
        "production persistence adapters remain disabled/deferred",
        "permanent consolidation commit APIs remain deferred behind explicit gates",
    ]

    return ReasoningReleaseAuditReport(
        stage=cfg.stage,
        source_pack=cfg.source_pack,
        module_availability=module_availability,
        public_import_availability=public_import_availability,
        disabled_default_verified=disabled_default_verified,
        no_mutation_verified=no_mutation_verified,
        config_serialization_verified=config_serialization_verified,
        trace_serialization_verified=trace_serialization_verified,
        optional_passes_verified=optional_passes_verified,
        remaining_deferred_work=remaining_deferred_work,
        readiness_level=readiness_level,
        metadata={"max_module_checks": cfg.max_module_checks},
    )


def reasoning_release_audit_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_release_audit",
        "stage": "REASON-2D",
        "non_mutating_audit": True,
        "checks_public_imports": True,
        "checks_json_serialization": True,
        "checks_disabled_default": True,
        "no_fake_production_complete_claim": True,
    }
