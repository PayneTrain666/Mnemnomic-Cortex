from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json

import torch

from .reasoning_controller import ReasoningController, ReasoningControllerConfig, ReasoningPassResult
from .reasoning_policy_router import ReasoningPolicyRouterConfig
from .evidence_reasoning_pass import EvidenceReasoningConfig
from .counterfactual_reasoning_probe import CounterfactualProbeConfig
from .conflict_aware_consolidation import ConflictAwareConsolidationConfig
from .reasoning_orchestration_trace import _safe_jsonable
from .multi_pass_thought_planner import MultiPassThoughtPlannerConfig, MultiPassThoughtPlanner
from .controller_planner_integration import ControllerPlannerIntegration, ControllerPlannerIntegrationConfig


class ReasoningControllerAPIError(ValueError):
    """Raised when the public reasoning controller API is used unsafely."""


@dataclass(frozen=True)
class ReasoningControllerAPIConfig:
    """Stable public config wrapper for the reasoning controller API.

    The API wrapper is disabled by default and preserves all REASON-2C
    no-mutation and opt-in guarantees. The allow_* flags control whether the
    wrapper may activate optional subpasses when building an enabled
    controller. Disallowed features are forced off even if the caller passes
    a serialized config attempting to enable them.
    """

    enabled: bool = False
    key_dim: int = 32
    value_dim: int = 32
    slot_count: int = 32
    max_reasoning_hops: int = 2
    allow_policy_router: bool = False
    allow_evidence_reasoning: bool = False
    allow_counterfactual_probe: bool = False
    allow_conflict_aware_consolidation: bool = False
    allow_multi_pass_planner: bool = False
    allow_controller_planner_integration: bool = False
    finite_checks: bool = True
    require_json_safe_outputs: bool = True
    no_mutation_by_default: bool = True

    def validate(self) -> None:
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise ReasoningControllerAPIError("key_dim and value_dim must be positive")
        if self.slot_count <= 0:
            raise ReasoningControllerAPIError("slot_count must be positive")
        if not (1 <= self.max_reasoning_hops <= 16):
            raise ReasoningControllerAPIError("max_reasoning_hops must be in [1,16]")
        if not self.no_mutation_by_default:
            raise ReasoningControllerAPIError("no_mutation_by_default must remain true for REASON-2D")

    @classmethod
    def disabled(cls, key_dim: int = 32, value_dim: Optional[int] = None) -> "ReasoningControllerAPIConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(cls, key_dim: int = 32, value_dim: Optional[int] = None, slot_count: int = 32) -> "ReasoningControllerAPIConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "slot_count": self.slot_count,
            "max_reasoning_hops": self.max_reasoning_hops,
            "allow_policy_router": self.allow_policy_router,
            "allow_evidence_reasoning": self.allow_evidence_reasoning,
            "allow_counterfactual_probe": self.allow_counterfactual_probe,
            "allow_conflict_aware_consolidation": self.allow_conflict_aware_consolidation,
            "allow_multi_pass_planner": self.allow_multi_pass_planner,
            "allow_controller_planner_integration": self.allow_controller_planner_integration,
            "finite_checks": self.finite_checks,
            "require_json_safe_outputs": self.require_json_safe_outputs,
            "no_mutation_by_default": self.no_mutation_by_default,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReasoningControllerAPIConfig":
        if not isinstance(payload, dict):
            raise ReasoningControllerAPIError("config payload must be a dict")
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: payload[key] for key in payload if key in allowed}
        config = cls(**clean)
        config.validate()
        return config


@dataclass
class ReasoningControllerAPIResult:
    """Stable JSON-safe public API result envelope."""

    result: ReasoningPassResult
    api_config: ReasoningControllerAPIConfig

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "api_config": self.api_config.to_dict(),
            "reasoning_result": self.result.to_dict(),
            "api_safety": {
                "no_permanent_memory_store_mutation": True,
                "no_model_weight_mutation": True,
                "no_optimizer_mutation": True,
                "optional_features_opt_in_only": True,
                "planner_opt_in_only": True,
                "canonical_slot_prefix": "reason2a.",
            },
        }
        return _safe_jsonable(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class ReasoningControllerAPI:
    """Public API wrapper around ReasoningController.

    This wrapper centralizes config serialization, disabled/enabled controller
    construction, JSON-safe outputs, and no-mutation enforcement.
    """

    def __init__(self, config: Optional[ReasoningControllerAPIConfig] = None):
        self.config = config or ReasoningControllerAPIConfig.disabled()
        self.config.validate()
        self.controller = self.build_controller(self.config)

    @staticmethod
    def build_controller(config: ReasoningControllerAPIConfig) -> ReasoningController:
        config.validate()
        # Underlying enabled MANN/LTM depth banks may default read_top_k_slots to 8.
        # Preserve the public API slot_count exactly in the serialized API config,
        # but provision the internal controller with at least 8 slots so small
        # smoke-test configs remain valid and no hidden runtime failure occurs.
        internal_slot_count = max(config.slot_count, 8) if config.enabled else config.slot_count
        controller_config = ReasoningControllerConfig(
            enabled=config.enabled,
            key_dim=config.key_dim,
            value_dim=config.value_dim,
            slot_count=internal_slot_count,
            max_reasoning_hops=config.max_reasoning_hops,
            finite_checks=config.finite_checks,
            no_mutation_by_default=True,
            use_policy_router=bool(config.enabled and config.allow_policy_router),
            policy_router_config=ReasoningPolicyRouterConfig.enabled_default() if config.enabled and config.allow_policy_router else None,
            use_evidence_reasoning=bool(config.enabled and config.allow_evidence_reasoning),
            evidence_config=EvidenceReasoningConfig.enabled_default() if config.enabled and config.allow_evidence_reasoning else None,
            use_counterfactual_probe=bool(config.enabled and config.allow_counterfactual_probe),
            counterfactual_config=CounterfactualProbeConfig.enabled_default() if config.enabled and config.allow_counterfactual_probe else None,
            use_conflict_aware_consolidation=bool(config.enabled and config.allow_conflict_aware_consolidation),
            conflict_config=ConflictAwareConsolidationConfig.enabled_default() if config.enabled and config.allow_conflict_aware_consolidation else None,
        )
        return ReasoningController(controller_config)

    @classmethod
    def from_serialized_config(cls, payload: Dict[str, Any]) -> "ReasoningControllerAPI":
        return cls(ReasoningControllerAPIConfig.from_dict(payload))

    def run_reasoning_pass(
        self,
        query: torch.Tensor,
        *,
        content: str = "",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        write_permission: bool = False,
    ) -> ReasoningControllerAPIResult:
        if write_permission:
            # REASON-2D API remains a release-readiness hardening layer, not a
            # permanent write enabler. Later stages may add explicit commit APIs.
            raise ReasoningControllerAPIError("write_permission=True is not allowed through REASON-2D public API")
        before = query.clone() if isinstance(query, torch.Tensor) else None
        result = self.controller.run_reasoning_pass(
            query,
            content=content,
            project_id=project_id,
            chat_id=chat_id,
            episode_id=episode_id,
            write_permission=False,
        )
        if before is not None and not torch.equal(query, before):
            raise ReasoningControllerAPIError("query tensor was mutated")
        if self.config.enabled and self.config.allow_multi_pass_planner:
            planner = MultiPassThoughtPlanner(MultiPassThoughtPlannerConfig.enabled_default())
            planner_report = planner.plan(query, content=content)
            result.trace.add_event(
                "multi_pass_thought_planner",
                "optional planner pass executed through API",
                planner_report.to_dict(),
            )
        if self.config.enabled and self.config.allow_controller_planner_integration:
            integration = ControllerPlannerIntegration(ControllerPlannerIntegrationConfig.enabled_default())
            integration_report = integration.run(query, content=content, write_permission=False, lineage={"api_stage": "REASON-3C"})
            result.trace.add_event(
                "controller_planner_integration",
                "optional planner quality integration executed through API",
                integration_report.to_dict(),
            )
        api_result = ReasoningControllerAPIResult(result=result, api_config=self.config)
        if self.config.require_json_safe_outputs:
            json.dumps(api_result.to_dict(), sort_keys=True)
        return api_result

    def contract_summary(self) -> Dict[str, Any]:
        return reasoning_controller_api_contract()


def reasoning_controller_api_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_controller_api",
        "stage": "REASON-2D",
        "default_enabled": False,
        "public_api_wrapper": True,
        "config_serialization": True,
        "json_safe_result": True,
        "write_permission_public_api": False,
        "permanent_memory_store_mutation": False,
        "canonical_slot_prefix_compatibility": "reason2a.",
        "optional_policy_router": "opt_in_only",
        "optional_evidence_reasoning": "opt_in_only",
        "optional_counterfactual_probe": "opt_in_only",
        "optional_conflict_aware_consolidation": "opt_in_only",
        "optional_multi_pass_planner": "opt_in_only",
        "optional_controller_planner_integration": "opt_in_only",
    }
