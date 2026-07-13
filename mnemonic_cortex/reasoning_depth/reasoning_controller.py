"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: reasoning controller.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib

import torch

from .wm_depth_controller import WMDepthController
from .mann_depth_adapter import MANNDepthAdapter, MANNDepthAdapterConfig
from .ltm_depth_adapter import LTMDepthAdapter, LTMDepthAdapterConfig
from .shared_depth_slot_registry import SharedDepthSlotRegistry
from .depth_integration_readiness import evaluate_depth_integration_readiness
from .depth_capacity_validation import DepthCapacityValidationConfig
from .reasoning_orchestration_trace import ReasoningOrchestrationTrace, _safe_jsonable
from .consolidation_gate import ShadowConsolidationGate, ConsolidationGateConfig
from .reasoning_policy_router import ReasoningPolicyRouter, ReasoningPolicyRouterConfig
from .evidence_reasoning_pass import EvidenceReasoningPass, EvidenceReasoningConfig
from .counterfactual_reasoning_probe import CounterfactualReasoningProbe, CounterfactualProbeConfig
from .conflict_aware_consolidation import ConflictAwareConsolidationEvaluator, ConflictAwareConsolidationConfig
from .mann_ltm_shared_slot_geometry import MANNLTMSharedSlotGeometry, SharedGeometrySlotConfig


class ReasoningControllerError(ValueError):
    """Raised when the reasoning controller receives invalid input."""


@dataclass(frozen=True)
class SharedGeometryRoutingPolicyConfig:
    """Routing policy for shared MANN/LTM geometry hops."""

    mann_slot_strategy: str = "hop_mod"
    ltm_slot_strategy: str = "hop_mod"
    ltm_depth_strategy: str = "fixed"
    fixed_mann_slot_index: int = 0
    fixed_ltm_slot_index: int = 0
    fixed_ltm_depth_index: int = 5
    mann_geometry_map: Optional[str] = None
    ltm_geometry_map: Optional[str] = None

    def validate(self) -> None:
        if self.mann_slot_strategy not in {"hop_mod", "fixed", "content_hash"}:
            raise ReasoningControllerError("invalid mann_slot_strategy")
        if self.ltm_slot_strategy not in {"hop_mod", "fixed", "content_hash"}:
            raise ReasoningControllerError("invalid ltm_slot_strategy")
        if self.ltm_depth_strategy not in {"fixed", "hop_mod"}:
            raise ReasoningControllerError("invalid ltm_depth_strategy")
        if self.fixed_mann_slot_index < 0:
            raise ReasoningControllerError("fixed_mann_slot_index must be >= 0")
        if self.fixed_ltm_slot_index < 0:
            raise ReasoningControllerError("fixed_ltm_slot_index must be >= 0")
        if not (0 <= self.fixed_ltm_depth_index < 8):
            raise ReasoningControllerError("fixed_ltm_depth_index must be in [0,7]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mann_slot_strategy": self.mann_slot_strategy,
            "ltm_slot_strategy": self.ltm_slot_strategy,
            "ltm_depth_strategy": self.ltm_depth_strategy,
            "fixed_mann_slot_index": self.fixed_mann_slot_index,
            "fixed_ltm_slot_index": self.fixed_ltm_slot_index,
            "fixed_ltm_depth_index": self.fixed_ltm_depth_index,
            "mann_geometry_map": self.mann_geometry_map,
            "ltm_geometry_map": self.ltm_geometry_map,
        }


@dataclass(frozen=True)
class ReasoningControllerConfig:
    """Configuration for the additive REASON-2C controller.

    Default behavior remains disabled/pass-through. Policy, evidence,
    counterfactual, and conflict-aware passes are all opt-in and non-mutating.
    """

    enabled: bool = False
    key_dim: int = 32
    value_dim: int = 32
    slot_count: int = 32
    max_reasoning_hops: int = 2
    max_trace_events: int = 96
    finite_checks: bool = True
    no_mutation_by_default: bool = True
    propose_ltm_consolidation: bool = True
    use_policy_router: bool = False
    policy_router_config: Optional[ReasoningPolicyRouterConfig] = None
    use_evidence_reasoning: bool = False
    evidence_config: Optional[EvidenceReasoningConfig] = None
    use_counterfactual_probe: bool = False
    counterfactual_config: Optional[CounterfactualProbeConfig] = None
    use_conflict_aware_consolidation: bool = False
    conflict_config: Optional[ConflictAwareConsolidationConfig] = None
    use_shared_mann_ltm_geometry: bool = False
    shared_geometry_config: Optional[SharedGeometrySlotConfig] = None
    shared_geometry_routing_policy: Optional[SharedGeometryRoutingPolicyConfig] = None

    def validate(self) -> None:
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise ReasoningControllerError("key_dim/value_dim must be positive")
        if self.slot_count <= 0:
            raise ReasoningControllerError("slot_count must be positive")
        if not (1 <= self.max_reasoning_hops <= 16):
            raise ReasoningControllerError("max_reasoning_hops must be in [1,16]")
        if not (8 <= self.max_trace_events <= 2048):
            raise ReasoningControllerError("max_trace_events must be in [8,2048]")
        if self.use_shared_mann_ltm_geometry and self.key_dim != self.value_dim:
            raise ReasoningControllerError("shared MANN/LTM geometry requires key_dim == value_dim")
        for cfg in (self.policy_router_config, self.evidence_config, self.counterfactual_config, self.conflict_config):
            if cfg is not None:
                cfg.validate()
        if self.shared_geometry_routing_policy is not None:
            self.shared_geometry_routing_policy.validate()

    @classmethod
    def disabled(cls, key_dim: int = 32, value_dim: Optional[int] = None) -> "ReasoningControllerConfig":
        return cls(enabled=False, key_dim=key_dim, value_dim=value_dim or key_dim)

    @classmethod
    def enabled_default(cls, key_dim: int = 32, value_dim: Optional[int] = None, slot_count: int = 32) -> "ReasoningControllerConfig":
        return cls(enabled=True, key_dim=key_dim, value_dim=value_dim or key_dim, slot_count=slot_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "slot_count": self.slot_count,
            "max_reasoning_hops": self.max_reasoning_hops,
            "max_trace_events": self.max_trace_events,
            "finite_checks": self.finite_checks,
            "no_mutation_by_default": self.no_mutation_by_default,
            "propose_ltm_consolidation": self.propose_ltm_consolidation,
            "use_policy_router": self.use_policy_router,
            "policy_router_config": self.policy_router_config.to_dict() if self.policy_router_config is not None else None,
            "use_evidence_reasoning": self.use_evidence_reasoning,
            "evidence_config": self.evidence_config.to_dict() if self.evidence_config is not None else None,
            "use_counterfactual_probe": self.use_counterfactual_probe,
            "counterfactual_config": self.counterfactual_config.to_dict() if self.counterfactual_config is not None else None,
            "use_conflict_aware_consolidation": self.use_conflict_aware_consolidation,
            "conflict_config": self.conflict_config.to_dict() if self.conflict_config is not None else None,
            "use_shared_mann_ltm_geometry": self.use_shared_mann_ltm_geometry,
            "shared_geometry_config": self.shared_geometry_config.to_dict() if self.shared_geometry_config is not None else None,
            "shared_geometry_routing_policy": self.shared_geometry_routing_policy.to_dict() if self.shared_geometry_routing_policy is not None else None,
        }


@dataclass
class ReasoningPassResult:
    """Output from one bounded reasoning pass."""

    output: torch.Tensor
    trace: ReasoningOrchestrationTrace
    wm_output: Optional[torch.Tensor] = None
    mann_output: Optional[torch.Tensor] = None
    ltm_output: Optional[torch.Tensor] = None
    consolidation_evaluation: Optional[Any] = None
    readiness: Optional[Dict[str, Any]] = None
    mutation_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_shape": list(self.output.shape),
            "wm_output_shape": list(self.wm_output.shape) if self.wm_output is not None else None,
            "mann_output_shape": list(self.mann_output.shape) if self.mann_output is not None else None,
            "ltm_output_shape": list(self.ltm_output.shape) if self.ltm_output is not None else None,
            "trace": self.trace.to_dict(),
            "consolidation_evaluation": self.consolidation_evaluation.to_dict() if self.consolidation_evaluation is not None else None,
            "readiness": _safe_jsonable(self.readiness),
            "mutation_performed": self.mutation_performed,
            "safety": {
                "permanent_memory_store_mutation": False,
                "destructive_replacement": False,
                "shadow_consolidation_only_by_default": True,
            },
        }


@dataclass
class ReasoningController:
    """Safe WM→MANN→LTM depth orchestration controller.

    REASON-2C adds optional evidence, counterfactual, and conflict-aware
    consolidation metadata. Default behavior is still disabled/pass-through.
    """

    config: ReasoningControllerConfig
    wm_controller: Optional[WMDepthController] = None
    mann_adapter: Optional[MANNDepthAdapter] = None
    ltm_adapter: Optional[LTMDepthAdapter] = None
    registry: Optional[SharedDepthSlotRegistry] = None
    consolidation_gate: Optional[ShadowConsolidationGate] = None
    policy_router: Optional[ReasoningPolicyRouter] = None
    evidence_pass: Optional[EvidenceReasoningPass] = None
    counterfactual_probe: Optional[CounterfactualReasoningProbe] = None
    conflict_evaluator: Optional[ConflictAwareConsolidationEvaluator] = None
    shared_geometry_orchestrator: Optional[MANNLTMSharedSlotGeometry] = None

    def __post_init__(self) -> None:
        self.config.validate()
        if self.wm_controller is None:
            self.wm_controller = WMDepthController.disabled(input_dim=self.config.key_dim)
        if self.mann_adapter is None:
            self.mann_adapter = (
                MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(self.config.key_dim, self.config.value_dim, self.config.slot_count))
                if self.config.enabled
                else MANNDepthAdapter(MANNDepthAdapterConfig.disabled(self.config.key_dim, self.config.value_dim))
            )
        if self.ltm_adapter is None:
            self.ltm_adapter = (
                LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(self.config.key_dim, self.config.value_dim, self.config.slot_count))
                if self.config.enabled
                else LTMDepthAdapter(LTMDepthAdapterConfig.disabled(self.config.key_dim, self.config.value_dim))
            )
        if self.registry is None:
            self.registry = SharedDepthSlotRegistry()
        if self.consolidation_gate is None:
            self.consolidation_gate = ShadowConsolidationGate(ConsolidationGateConfig())
        if self.policy_router is None:
            router_cfg = self.config.policy_router_config or ReasoningPolicyRouterConfig(enabled=self.config.use_policy_router)
            self.policy_router = ReasoningPolicyRouter(router_cfg)
        if self.evidence_pass is None:
            evidence_cfg = self.config.evidence_config or EvidenceReasoningConfig(enabled=self.config.use_evidence_reasoning)
            self.evidence_pass = EvidenceReasoningPass(evidence_cfg)
        if self.counterfactual_probe is None:
            cf_cfg = self.config.counterfactual_config or CounterfactualProbeConfig(enabled=self.config.use_counterfactual_probe)
            self.counterfactual_probe = CounterfactualReasoningProbe(cf_cfg)
        if self.conflict_evaluator is None:
            conflict_cfg = self.config.conflict_config or ConflictAwareConsolidationConfig(enabled=self.config.use_conflict_aware_consolidation)
            self.conflict_evaluator = ConflictAwareConsolidationEvaluator(conflict_cfg)
        if self.shared_geometry_orchestrator is None:
            geometry_cfg = self.config.shared_geometry_config or SharedGeometrySlotConfig(
                enabled=self.config.use_shared_mann_ltm_geometry,
                key_dim=self.config.key_dim,
                value_dim=self.config.value_dim,
            )
            self.shared_geometry_orchestrator = MANNLTMSharedSlotGeometry(
                config=geometry_cfg,
                mann_adapter=self.mann_adapter,
                ltm_adapter=self.ltm_adapter,
                registry=self.registry,
            )

    def _shared_routing_policy(self) -> SharedGeometryRoutingPolicyConfig:
        return self.config.shared_geometry_routing_policy or SharedGeometryRoutingPolicyConfig()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def run_reasoning_pass(
        self,
        query: torch.Tensor,
        *,
        content: str = "",
        project_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        write_permission: bool = False,
        return_trace: bool = True,
    ) -> ReasoningPassResult:
        self._validate_query(query)
        trace = ReasoningOrchestrationTrace(max_events=self.config.max_trace_events)
        trace.metadata.update({"controller_enabled": self.enabled, "project_id": project_id, "chat_id": chat_id, "episode_id": episode_id, "write_permission": write_permission})

        if not self.enabled:
            trace.add_event("reasoning_controller", "controller disabled; pass-through returned", {"input_shape": list(query.shape), "no_mutation": True})
            return ReasoningPassResult(output=query, trace=trace, wm_output=query, mutation_performed=False)

        trace.add_event("reasoning_controller", "enabled bounded reasoning pass started", {"input_shape": list(query.shape)})

        effective_hops = self.config.max_reasoning_hops
        ltm_bank_name = "cgmn_semantic"
        gate_confidence = 0.75
        gate_disagreement = 0.0
        if self.config.use_policy_router and self.policy_router is not None:
            policy_decision = self.policy_router.route(query, content=content, metadata={"source_stage": "REASON-2C"})
            trace.add_event("reasoning_policy_router", "policy route selected", policy_decision.to_dict())
            effective_hops = min(self.config.max_reasoning_hops, policy_decision.route_plan.max_hops)
            ltm_bank_name = policy_decision.route_plan.preferred_ltm_bank
            gate_confidence = policy_decision.score_report.confidence
            gate_disagreement = policy_decision.score_report.disagreement

        evidence_report = self.evidence_pass.run(content=content, query=query, metadata={"source_stage": "REASON-2C"}) if self.evidence_pass is not None else None
        if evidence_report is not None and evidence_report.enabled:
            trace.add_event("evidence_reasoning_pass", "evidence reasoning report created", evidence_report.to_dict())

        counterfactual_report = None
        if self.counterfactual_probe is not None and evidence_report is not None:
            counterfactual_report = self.counterfactual_probe.run(
                evidence_report,
                base_confidence=gate_confidence,
                base_disagreement=gate_disagreement,
                metadata={"source_stage": "REASON-2C"},
            )
            if counterfactual_report.enabled:
                trace.add_event("counterfactual_reasoning_probe", "counterfactual probe report created", counterfactual_report.to_dict())

        conflict_report = None
        if self.conflict_evaluator is not None:
            conflict_report = self.conflict_evaluator.evaluate(
                evidence_report=evidence_report,
                counterfactual_report=counterfactual_report,
                confidence=gate_confidence,
                disagreement=gate_disagreement,
                base_conflict=False,
                metadata={"source_stage": "REASON-2C"},
            )
            if conflict_report.enabled:
                trace.add_event("conflict_aware_consolidation", "conflict-aware consolidation report created", conflict_report.to_dict())
                gate_confidence = conflict_report.adjusted_confidence
                gate_disagreement = conflict_report.adjusted_disagreement

        wm_out, wm_trace = self.wm_controller.process_wm(query, return_trace=True)
        trace.add_event("wm_depth_controller", "WM depth controller processed query", wm_trace)

        mann_out = None
        ltm_out = None
        current = wm_out
        if self.config.use_shared_mann_ltm_geometry:
            routing_policy = self._shared_routing_policy()
            for hop in range(effective_hops):
                mann_slot_index = self._resolve_shared_slot_index(
                    strategy=routing_policy.mann_slot_strategy,
                    fixed_index=routing_policy.fixed_mann_slot_index,
                    hop=hop,
                    content=content,
                )
                ltm_slot_index = self._resolve_shared_slot_index(
                    strategy=routing_policy.ltm_slot_strategy,
                    fixed_index=routing_policy.fixed_ltm_slot_index,
                    hop=hop,
                    content=content,
                )
                ltm_depth_index = self._resolve_ltm_depth_index(
                    strategy=routing_policy.ltm_depth_strategy,
                    fixed_depth=routing_policy.fixed_ltm_depth_index,
                    hop=hop,
                )
                shared_out, shared_trace = self.shared_geometry_orchestrator.run_shared_reasoning(
                    current,
                    content=content or "reasoning_pass",
                    mann_slot_index=mann_slot_index,
                    ltm_slot_index=ltm_slot_index,
                    hop_id=hop,
                    ltm_bank_name=ltm_bank_name,
                    ltm_depth_index=ltm_depth_index,
                    mann_geometry_map=routing_policy.mann_geometry_map,
                    ltm_geometry_map=routing_policy.ltm_geometry_map,
                    canonical_slot_id=f"{self._canonical_slot_id(content or 'reasoning_pass', project_id, chat_id, episode_id)}.h{hop}",
                    return_trace=True,
                )
                trace.add_event("mann_ltm_shared_slot_geometry", f"shared geometry hop {hop} completed", shared_trace)
                current = shared_out.unsqueeze(1)
            mann_out = current.squeeze(1)
            ltm_out = mann_out
        else:
            for hop in range(effective_hops):
                mann_out, mann_trace = self.mann_adapter.read_hop(current, hop_id=hop, return_trace=True)
                trace.add_event("mann_depth_adapter", f"MANN hop {hop} completed", mann_trace)
                current = mann_out.unsqueeze(1)

            ltm_out, ltm_trace = self.ltm_adapter.read_ltm(mann_out, bank_name=ltm_bank_name, return_trace=True)
            trace.add_event("ltm_depth_adapter", "LTM depth read completed", ltm_trace)

        readiness = evaluate_depth_integration_readiness(
            DepthCapacityValidationConfig(
                slot_counts=(min(self.config.slot_count, 16),),
                key_dim=self.config.key_dim,
                value_dim=self.config.value_dim,
                max_smoke_batch=min(2, query.shape[0]),
                max_smoke_tokens=min(3, query.shape[1] if query.dim() == 3 else 1),
            )
        ).to_dict()
        trace.add_event("integration_readiness", "readiness snapshot captured", {"readiness_level": readiness["readiness_level"]})

        consolidation_eval = None
        if self.config.propose_ltm_consolidation:
            canonical_slot_id = self._canonical_slot_id(content or "reasoning_pass", project_id, chat_id, episode_id)
            proposal = self.ltm_adapter.propose_consolidation(
                canonical_slot_id=canonical_slot_id,
                content=content or "reasoning_pass",
                bank_name=ltm_bank_name,
                slot_index=0,
                depth_index=5,
                value=ltm_out[0].detach().clone(),
                key=mann_out[0].detach().clone(),
                source_stage="REASON-2C",
                source_pack="reasoning_controller",
                project_id=project_id,
                chat_id=chat_id,
                episode_id=episode_id,
            )
            trace.add_event("ltm_consolidation_proposal", "shadow LTM consolidation proposal created", {"canonical_slot_id": canonical_slot_id})
            consolidation_eval = self.consolidation_gate.evaluate(
                proposal,
                write_permission=write_permission,
                confidence=gate_confidence,
                disagreement=gate_disagreement,
                conflict=bool(conflict_report.quarantine_recommended) if conflict_report is not None else False,
                canonical_slot_id=canonical_slot_id,
            )
            trace.add_event("consolidation_gate", "consolidation gate evaluated proposal", consolidation_eval.to_dict())

        output = (mann_out + ltm_out) / 2.0
        if self.config.finite_checks and not torch.isfinite(output).all():
            raise ReasoningControllerError("reasoning output contains NaN/Inf")
        trace.add_event("reasoning_controller", "bounded reasoning pass completed", {"output_shape": list(output.shape)})

        return ReasoningPassResult(
            output=output,
            trace=trace,
            wm_output=wm_out,
            mann_output=mann_out,
            ltm_output=ltm_out,
            consolidation_evaluation=consolidation_eval,
            readiness=readiness,
            mutation_performed=False,
        )

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise ReasoningControllerError("query must be a torch.Tensor")
        if query.dim() not in {2, 3}:
            raise ReasoningControllerError("query must be [B,D] or [B,T,D]")
        if query.size(-1) != self.config.key_dim:
            raise ReasoningControllerError(f"query last dim must be {self.config.key_dim}")
        if self.config.finite_checks and not torch.isfinite(query).all():
            raise ReasoningControllerError("query contains NaN/Inf")

    def _canonical_slot_id(self, content: str, project_id: Optional[str], chat_id: Optional[str], episode_id: Optional[str]) -> str:
        # Preserve REASON-2A/2B canonical ID prefix for backward compatibility.
        seed = "|".join([project_id or "", chat_id or "", episode_id or "", content])
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"reason2a.{digest}"

    def _resolve_shared_slot_index(self, *, strategy: str, fixed_index: int, hop: int, content: str) -> int:
        if strategy == "fixed":
            return int(fixed_index % max(1, self.config.slot_count))
        if strategy == "content_hash":
            content_digest = hashlib.sha256((content or "reasoning_pass").encode("utf-8")).hexdigest()
            return int(content_digest[:8], 16) % max(1, self.config.slot_count)
        return int(hop % max(1, self.config.slot_count))

    def _resolve_ltm_depth_index(self, *, strategy: str, fixed_depth: int, hop: int) -> int:
        if strategy == "hop_mod":
            return int(hop % 8)
        return int(fixed_depth)


def reasoning_controller_contract() -> Dict[str, Any]:
    return {
        "module": "reasoning_controller",
        "stage": "REASON-2C",
        "default_enabled": False,
        "orchestrates": [
            "WMDepthController",
            "MANNDepthAdapter",
            "LTMDepthAdapter",
            "SharedDepthSlotRegistry",
            "ShadowConsolidationGate",
            "ReasoningPolicyRouter",
            "EvidenceReasoningPass",
            "CounterfactualReasoningProbe",
            "ConflictAwareConsolidationEvaluator",
        ],
        "permanent_memory_store_mutation": False,
        "shadow_consolidation_only_by_default": True,
        "bounded_hops": True,
        "trace_serialization": True,
        "optional_policy_router": True,
        "optional_evidence_reasoning": True,
        "optional_counterfactual_probe": True,
        "optional_conflict_aware_consolidation": True,
        "optional_shared_mann_ltm_geometry": True,
        "shared_geometry_routing_policy": {
            "mann_slot_strategy": ["hop_mod", "fixed", "content_hash"],
            "ltm_slot_strategy": ["hop_mod", "fixed", "content_hash"],
            "ltm_depth_strategy": ["fixed", "hop_mod"],
        },
        "canonical_slot_prefix_compatibility": "reason2a.",
    }
