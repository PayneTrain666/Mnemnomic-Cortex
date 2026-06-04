import json
import torch

from mnemonic_cortex.reasoning_depth import (
    ReasoningController,
    ReasoningControllerConfig,
    ReasoningPolicyRouterConfig,
    EvidenceReasoningConfig,
    CounterfactualProbeConfig,
    ConflictAwareConsolidationConfig,
)


def test_reason2c_controller_optional_evidence_counterfactual_conflict_trace():
    cfg = ReasoningControllerConfig(
        enabled=True,
        key_dim=16,
        value_dim=16,
        slot_count=8,
        use_policy_router=True,
        policy_router_config=ReasoningPolicyRouterConfig.enabled_default(task_mode="hypothesis"),
        use_evidence_reasoning=True,
        evidence_config=EvidenceReasoningConfig.enabled_default(),
        use_counterfactual_probe=True,
        counterfactual_config=CounterfactualProbeConfig.enabled_default(),
        use_conflict_aware_consolidation=True,
        conflict_config=ConflictAwareConsolidationConfig.enabled_default(),
    )
    controller = ReasoningController(cfg)
    x = torch.randn(2, 3, 16)
    before = x.clone()
    result = controller.run_reasoning_pass(x, content="Evidence one. Evidence two.", project_id="p", chat_id="c", episode_id="e")
    payload = result.to_dict()
    stages = [event["stage"] for event in payload["trace"]["events"]]

    assert torch.equal(x, before)
    assert "evidence_reasoning_pass" in stages
    assert "counterfactual_reasoning_probe" in stages
    assert "conflict_aware_consolidation" in stages
    assert payload["mutation_performed"] is False
    json.dumps(payload)
