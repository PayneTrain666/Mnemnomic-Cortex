import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig


def test_reason2d_controller_api_enabled_bounded_pass():
    cfg = ReasoningControllerAPIConfig(
        enabled=True,
        key_dim=8,
        value_dim=8,
        slot_count=8,
        allow_policy_router=True,
        allow_evidence_reasoning=True,
        allow_counterfactual_probe=True,
        allow_conflict_aware_consolidation=True,
    )
    api = ReasoningControllerAPI(cfg)
    result = api.run_reasoning_pass(torch.randn(1, 2, 8), content="Evidence. More evidence.")
    payload = result.to_dict()
    stages = [event["stage"] for event in payload["reasoning_result"]["trace"]["events"]]

    assert "reasoning_policy_router" in stages
    assert "evidence_reasoning_pass" in stages
    assert payload["api_safety"]["no_permanent_memory_store_mutation"] is True
