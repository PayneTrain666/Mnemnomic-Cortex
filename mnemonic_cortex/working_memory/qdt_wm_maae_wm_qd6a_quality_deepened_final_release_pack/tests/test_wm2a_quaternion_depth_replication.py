import math

import torch

from mnemonic_cortex.working_memory import (
    QuaternionDepthConfig,
    QuaternionDepthReplicator,
    normalize_quaternion,
    quaternion_multiply,
    rotate_vectors_by_quaternion,
)


def test_normalize_quaternion_unit_norm():
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    qn = normalize_quaternion(q)
    assert torch.allclose(qn.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_quaternion_multiply_identity():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = normalize_quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    out = quaternion_multiply(identity, q)
    assert torch.allclose(out, q, atol=1e-6)


def test_rotate_vectors_by_quaternion_90deg_z():
    angle = math.pi / 2.0
    q = torch.tensor([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = rotate_vectors_by_quaternion(v, q)
    expected = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(rotated, expected, atol=1e-5)


def test_quaternion_depth_replicator_output_shape_and_trace():
    rep = QuaternionDepthReplicator(dim=9, num_depths=8)
    x = torch.randn(2, 5, 9)
    out, trace = rep(x, return_trace=True)

    assert out.shape == (2, 8, 5, 3, 9)
    assert trace["num_depths"] == 8
    assert trace["triplet_dim"] == 3
    assert trace["full_3d_blocks"] == 3
    assert trace["remainder_dim"] == 0
    assert abs(trace["quaternion_norm_min"] - 1.0) < 1e-5
    assert abs(trace["quaternion_norm_max"] - 1.0) < 1e-5


def test_quaternion_depth_replicator_preserves_remainder_dims():
    rep = QuaternionDepthReplicator(dim=10, num_depths=8)
    x = torch.randn(2, 5, 10)
    out = rep(x)

    assert out.shape == (2, 8, 5, 3, 10)
    original_remainder = x[..., -1:]
    replicated_remainder = out[..., -1:]
    target = original_remainder.unsqueeze(1).unsqueeze(3).expand_as(replicated_remainder)
    assert torch.allclose(replicated_remainder, target, atol=1e-5)

    report = rep.depth_consistency_report(x)
    assert report["ok"] is True
    assert report["remainder_ok"] is True


def test_quaternion_depth_replicator_applies_real_rotation():
    rep = QuaternionDepthReplicator(dim=3, num_depths=1)
    angle = math.pi / 2.0
    q = torch.tensor([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])
    rep.set_depth_quaternion(0, 0, q)

    x = torch.tensor([[[1.0, 0.0, 0.0]]])
    out = rep(x)

    # Anchor triplet at depth 0 should rotate x-axis to y-axis.
    anchor = out[0, 0, 0, 0]
    assert torch.allclose(anchor, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_quaternion_depth_config_path_and_dual_status():
    cfg = QuaternionDepthConfig(dim=11, num_depths=4)
    rep = QuaternionDepthReplicator(config=cfg)
    x = torch.randn(2, 3, 11)
    out = rep(x)
    assert out.shape == (2, 4, 3, 3, 11)
    status = rep.dual_quaternion_status()
    assert status["available"] is False
    assert status["status"] == "placeholder_not_implemented"


def test_quaternion_depth_rejects_bad_shape():
    rep = QuaternionDepthReplicator(dim=9, num_depths=8)
    bad = torch.randn(2, 9)
    try:
        rep(bad)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
