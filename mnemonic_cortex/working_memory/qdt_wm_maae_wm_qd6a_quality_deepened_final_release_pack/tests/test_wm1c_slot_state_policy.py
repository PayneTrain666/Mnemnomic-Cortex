import torch

from mnemonic_cortex.working_memory import (
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    CurvatureMetricPolicyConfig,
    CurvatureMetricPolicy,
)


def test_curved_slot_state_bank_shapes_and_snapshot():
    cfg = CurvedSlotStateConfig(num_slots=6, dim=16)
    bank = CurvedSlotStateBank(cfg)
    snap = bank.snapshot()

    assert snap.content.shape == (6, 16)
    assert snap.position.shape == (6, 16)
    assert snap.tangent.shape == (6, 16)
    assert snap.phase.shape == (6, 16)
    assert snap.curvature.shape == (6,)
    assert snap.importance.shape == (6,)
    assert snap.confidence.shape == (6,)
    assert snap.last_updated.shape == (6,)
    assert len(snap.slot_id) == 6
    assert bank.validate_state()["ok"] is True


def test_curved_slot_state_update_repairs_and_traces():
    cfg = CurvedSlotStateConfig(num_slots=6, dim=16)
    bank = CurvedSlotStateBank(cfg)
    idx = torch.tensor([1, 3])
    new_position = torch.randn(2, 16) * 100.0
    new_curvature = torch.tensor([-99.0, 99.0])
    trace = bank.update_slots(
        idx,
        position=new_position,
        curvature=new_curvature,
        importance=torch.tensor([-1.0, 2.0]),
        confidence=torch.tensor([-1.0, 2.0]),
        trace_link="unit-test",
    )

    assert trace.operation == "update_slots"
    assert trace.metadata["ok"] is True
    snap = bank.snapshot(idx.tolist())
    assert torch.all(snap.position.norm(dim=-1) <= cfg.position_radius + 1e-6)
    assert torch.all(snap.curvature >= cfg.curvature_min)
    assert torch.all(snap.curvature <= cfg.curvature_max)
    assert torch.all(snap.importance >= 0.0)
    assert torch.all(snap.importance <= 1.0)
    assert torch.all(snap.confidence >= 0.0)
    assert torch.all(snap.confidence <= 1.0)
    assert "unit-test" in snap.trace_links[0]


def test_curvature_metric_policy_shapes_with_context():
    cfg = CurvatureMetricPolicyConfig(num_slots=6, num_depths=8, context_dim=16)
    policy = CurvatureMetricPolicy(cfg)
    context = torch.randn(2, 4, 16)
    out = policy(context=context)

    assert out.global_curvature.shape == (1,)
    assert out.per_slot_curvature.shape == (6,)
    assert out.per_depth_curvature.shape == (8,)
    assert out.context_curvature.shape == (2, 1)
    assert out.combined_curvature.shape == (2, 8, 6)
    assert out.trace["combined_shape"] == [2, 8, 6]
    assert "paamax_metadata" in out.trace


def test_curvature_metric_policy_clamps_and_repairs():
    cfg = CurvatureMetricPolicyConfig(num_slots=6, num_depths=8, context_dim=16, curvature_min=-1.0, curvature_max=1.0)
    policy = CurvatureMetricPolicy(cfg)
    with torch.no_grad():
        policy.global_curvature.fill_(10.0)
        policy.per_slot_curvature.fill_(10.0)
        policy.per_depth_curvature.fill_(-10.0)

    report = policy.repair_in_place()
    assert report["ok"] is True
    out = policy(batch_size=3)
    assert out.combined_curvature.shape == (3, 8, 6)
    assert torch.all(out.combined_curvature <= 1.0)
    assert torch.all(out.combined_curvature >= -1.0)


def test_curvature_metric_policy_reference_and_drift_penalty():
    cfg = CurvatureMetricPolicyConfig(num_slots=6, num_depths=8, context_dim=16)
    policy = CurvatureMetricPolicy(cfg)
    policy.set_reference_to_current()
    p0 = policy.drift_penalty()
    with torch.no_grad():
        policy.per_slot_curvature.add_(0.5)
    p1 = policy.drift_penalty()
    assert p1 > p0
