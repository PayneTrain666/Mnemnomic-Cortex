import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2a_ltm_shadow_proposal_routing_exists():
    controller = ReasoningController(ReasoningControllerConfig.enabled_default(key_dim=16, value_dim=16, slot_count=8))
    result = controller.run_reasoning_pass(
        torch.randn(2, 3, 16),
        content="ltm proposal routing",
        project_id="proj",
        chat_id="chat",
        episode_id="ep",
    )
    assert result.consolidation_evaluation is not None
    evaluation = result.consolidation_evaluation.to_dict()
    assert evaluation["committed"] is False
    assert evaluation["shadow_only"] is True
    assert evaluation["canonical_slot_id"].startswith("reason2a.")
