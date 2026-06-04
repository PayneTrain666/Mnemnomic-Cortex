import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2a_mann_hop_routing_respects_max_hops():
    controller = ReasoningController(
        ReasoningControllerConfig(enabled=True, key_dim=16, value_dim=16, slot_count=8, max_reasoning_hops=3)
    )
    result = controller.run_reasoning_pass(torch.randn(2, 4, 16), content="hop routing")
    trace = result.trace.to_dict()
    hop_events = [event for event in trace["events"] if event["stage"] == "mann_depth_adapter"]
    assert len(hop_events) == 3
    assert all("MANN hop" in event["message"] for event in hop_events)
