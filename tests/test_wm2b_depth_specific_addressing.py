import torch

from mnemonic_cortex.working_memory import (
    QuaternionDepthReplicator,
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    CurvatureMetricPolicyConfig,
    CurvatureMetricPolicy,
    DepthSpecificAddressingConfig,
    DepthSpecificAddressing,
)


def make_addressing(dim=32, slots=6, depths=8):
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=slots, dim=dim))
    policy = CurvatureMetricPolicy(CurvatureMetricPolicyConfig(num_slots=slots, num_depths=depths, context_dim=dim))
    addressing = DepthSpecificAddressing(
        DepthSpecificAddressingConfig(dim=dim, num_slots=slots, num_depths=depths, top_k=3),
        slot_bank=bank,
        curvature_policy=policy,
        default_context_map="spatial_mechanical",
    )
    return addressing, bank, policy


def test_depth_specific_addressing_shapes_and_trace():
    addressing, _, _ = make_addressing()
    depth_state = torch.randn(2, 8, 5, 3, 32)
    context = torch.randn(2, 4, 32)

    out, trace = addressing(depth_state, context=context, context_map_name="spatial_mechanical", return_trace=True)

    assert out.activation.shape == (2, 8, 6)
    assert out.scores.shape == (2, 8, 6)
    assert out.read_content.shape == (2, 8, 32)
    assert out.top_indices.shape == (2, 8, 3)
    assert trace["finite"] is True
    assert trace["activation_shape"] == [2, 8, 6]
    assert len(trace["geometry_by_depth"]) == 8
    assert "paamax_metadata" in trace
    assert torch.allclose(out.activation.sum(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_depth_specific_addressing_uses_curvature_bias():
    addressing, _, policy = make_addressing()
    depth_state = torch.randn(2, 8, 5, 3, 32)
    context = torch.randn(2, 4, 32)

    out1 = addressing(depth_state, context=context, context_map_name="literal")
    with torch.no_grad():
        policy.per_depth_curvature[0] = 5.0
        policy.per_slot_curvature[0] = 5.0
    out2 = addressing(depth_state, context=context, context_map_name="literal")

    # The curvature change should alter the depth/slot score distribution.
    assert not torch.allclose(out1.scores[:, 0, :], out2.scores[:, 0, :])


def test_depth_specific_addressing_is_compatible_with_quaternion_replicator():
    rep = QuaternionDepthReplicator(dim=32, num_depths=8)
    addressing, _, _ = make_addressing()

    x = torch.randn(2, 5, 32)
    depth_state = rep(x)
    out = addressing(depth_state, context_map_name="quantum_holographic")

    assert depth_state.shape == (2, 8, 5, 3, 32)
    assert out.activation.shape == (2, 8, 6)
    assert torch.isfinite(out.activation).all()
    assert torch.isfinite(out.read_content).all()


def test_depth_specific_addressing_rejects_bad_shape():
    addressing, _, _ = make_addressing()
    bad = torch.randn(2, 5, 32)
    try:
        addressing(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_depth_specific_addressing_stability_report():
    addressing, _, _ = make_addressing()
    depth_state = torch.randn(2, 8, 5, 3, 32)
    report = addressing.stability_report(depth_state)

    assert report["ok"] is True
    assert report["finite"] is True
    assert report["activation_shape"] == [2, 8, 6]
