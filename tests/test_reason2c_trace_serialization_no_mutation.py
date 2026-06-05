import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig, EvidenceReasoningConfig


def test_reason2c_trace_serialization_and_no_mutation():
    cfg = ReasoningControllerConfig(
        enabled=True,
        key_dim=16,
        value_dim=16,
        slot_count=8,
        use_evidence_reasoning=True,
        evidence_config=EvidenceReasoningConfig.enabled_default(),
    )
    controller = ReasoningController(cfg)
    x = torch.randn(2, 3, 16)
    before = x.clone()
    result = controller.run_reasoning_pass(x, content="Serialize this evidence.")
    assert torch.equal(x, before)
    payload = result.to_dict()
    assert payload["safety"]["permanent_memory_store_mutation"] is False
    json.dumps(payload)
