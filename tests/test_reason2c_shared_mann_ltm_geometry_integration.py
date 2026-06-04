import torch

from mnemonic_cortex.reasoning_depth import (
    ReasoningController,
    ReasoningControllerConfig,
    ReasoningControllerAPI,
    ReasoningControllerAPIConfig,
    SharedGeometryRoutingPolicyConfig,
)


def test_reason2c_controller_shared_mann_ltm_geometry_opt_in():
    controller = ReasoningController(
        ReasoningControllerConfig(
            enabled=True,
            key_dim=16,
            value_dim=16,
            slot_count=8,
            max_reasoning_hops=2,
            use_shared_mann_ltm_geometry=True,
        )
    )
    result = controller.run_reasoning_pass(torch.randn(2, 3, 16), content="shared geometry opt in")
    payload = result.to_dict()
    stages = [event["stage"] for event in payload["trace"]["events"]]

    assert "mann_ltm_shared_slot_geometry" in stages
    assert "ltm_depth_adapter" not in stages
    assert payload["mutation_performed"] is False
    assert payload["output_shape"] == [2, 16]


def test_reason2d_api_shared_mann_ltm_geometry_opt_in():
    api = ReasoningControllerAPI(
        ReasoningControllerAPIConfig(
            enabled=True,
            key_dim=8,
            value_dim=8,
            slot_count=8,
            allow_shared_mann_ltm_geometry=True,
        )
    )
    result = api.run_reasoning_pass(torch.randn(1, 2, 8), content="api shared geometry")
    payload = result.to_dict()
    stages = [event["stage"] for event in payload["reasoning_result"]["trace"]["events"]]

    assert "mann_ltm_shared_slot_geometry" in stages
    assert payload["api_config"]["allow_shared_mann_ltm_geometry"] is True


def test_reason2c_controller_shared_geometry_fixed_slot_and_depth_policy():
    controller = ReasoningController(
        ReasoningControllerConfig(
            enabled=True,
            key_dim=16,
            value_dim=16,
            slot_count=8,
            max_reasoning_hops=1,
            use_shared_mann_ltm_geometry=True,
            shared_geometry_routing_policy=SharedGeometryRoutingPolicyConfig(
                mann_slot_strategy="fixed",
                ltm_slot_strategy="fixed",
                ltm_depth_strategy="fixed",
                fixed_mann_slot_index=3,
                fixed_ltm_slot_index=6,
                fixed_ltm_depth_index=7,
            ),
        )
    )
    result = controller.run_reasoning_pass(torch.randn(1, 2, 16), content="fixed-policy")
    payload = result.to_dict()
    shared_events = [event for event in payload["trace"]["events"] if event["stage"] == "mann_ltm_shared_slot_geometry"]
    assert len(shared_events) == 1
    hop_payload = shared_events[0]["payload"]
    assert hop_payload["mann_ref"] == "mann.slot3.z0"
    assert hop_payload["ltm_ref"] == "ltm.cgmn_semantic.slot6.z7"


def test_reason2d_api_shared_geometry_content_hash_policy_is_deterministic():
    cfg = ReasoningControllerAPIConfig(
        enabled=True,
        key_dim=8,
        value_dim=8,
        slot_count=8,
        allow_shared_mann_ltm_geometry=True,
        shared_mann_ltm_geometry_routing_policy={
            "mann_slot_strategy": "content_hash",
            "ltm_slot_strategy": "content_hash",
            "ltm_depth_strategy": "hop_mod",
        },
    )
    api = ReasoningControllerAPI(cfg)
    first = api.run_reasoning_pass(torch.randn(1, 2, 8), content="stable-hash-policy").to_dict()
    second = api.run_reasoning_pass(torch.randn(1, 2, 8), content="stable-hash-policy").to_dict()
    first_payload = [event for event in first["reasoning_result"]["trace"]["events"] if event["stage"] == "mann_ltm_shared_slot_geometry"][0]["payload"]
    second_payload = [event for event in second["reasoning_result"]["trace"]["events"] if event["stage"] == "mann_ltm_shared_slot_geometry"][0]["payload"]

    assert first_payload["mann_ref"] == second_payload["mann_ref"]
    assert first_payload["ltm_ref"] == second_payload["ltm_ref"]
