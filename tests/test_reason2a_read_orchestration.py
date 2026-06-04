import json
import torch

from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig


def test_reason2a_enabled_read_orchestration_shapes_and_trace():
    controller = ReasoningController(ReasoningControllerConfig.enabled_default(key_dim=16, value_dim=16, slot_count=8))
    x = torch.randn(2, 3, 16)
    result = controller.run_reasoning_pass(x, content="read orchestration", project_id="p", chat_id="c", episode_id="e")

    assert result.output.shape == (2, 16)
    assert result.wm_output.shape == (2, 3, 16)
    assert result.mann_output.shape == (2, 16)
    assert result.ltm_output.shape == (2, 16)
    assert torch.isfinite(result.output).all()
    trace = result.trace.to_dict()
    stages = [event["stage"] for event in trace["events"]]
    assert "wm_depth_controller" in stages
    assert "mann_depth_adapter" in stages
    assert "ltm_depth_adapter" in stages
    assert trace["paamax_metadata"]["trace_governance"] is True
    json.dumps(result.to_dict())
