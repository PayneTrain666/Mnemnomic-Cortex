import json
import torch

from mnemonic_cortex.reasoning_depth import WMDepthController


def test_reason1b_context_candidate_routes_to_z3_z5_z7_shadow_proposals():
    controller = WMDepthController.enabled_default(input_dim=16, value_dim=16, slot_count=16)
    context = torch.randn(1, 6, 16)
    response = torch.randn(1, 4, 16)

    result = controller.route_context_candidate(
        context=context,
        response=response,
        candidate={"summary": "test context"},
        project_id="mnemonic",
        chat_id="chat.1",
        episode_id="ep.1",
    )

    assert result["committed"] is False
    assert result["shadow_only"] is True
    assert result["depth_routes"] == {
        "contextual_binding": 3,
        "temporal_episode": 5,
        "volatile_trace": 7,
    }
    assert set(result["proposals"].keys()) == {"contextual_binding", "temporal_episode", "volatile_trace"}
    assert result["paamax_metadata"]["write_permission_granted"] is False
    json.dumps(result)
