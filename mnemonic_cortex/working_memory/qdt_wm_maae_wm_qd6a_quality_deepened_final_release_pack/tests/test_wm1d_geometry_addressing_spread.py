import torch

from mnemonic_cortex.working_memory import (
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    CurvatureMetricPolicyConfig,
    CurvatureMetricPolicy,
    GeometryAwareAddressingConfig,
    GeometryAwareAddressing,
    BoundedAssociativeSpreadConfig,
    BoundedAssociativeSpread,
    CurvedResonanceConfig,
    CurvedResonantWMCore,
)


def make_bank(dim=16, slots=6):
    cfg = CurvedSlotStateConfig(num_slots=slots, dim=dim)
    bank = CurvedSlotStateBank(cfg)
    with torch.no_grad():
        bank.importance.fill_(0.5)
        bank.confidence.fill_(0.75)
    return bank


def test_geometry_aware_addressing_shapes_and_trace():
    bank = make_bank()
    policy = CurvatureMetricPolicy(CurvatureMetricPolicyConfig(num_slots=6, num_depths=8, context_dim=16))
    addressing = GeometryAwareAddressing(
        GeometryAwareAddressingConfig(dim=16, num_slots=6, num_depths=8, top_k=3),
        slot_bank=bank,
        curvature_policy=policy,
    )

    query = torch.randn(2, 16)
    context = torch.randn(2, 4, 16)
    out = addressing(query, context=context)

    assert out.activation.shape == (2, 6)
    assert out.scores.shape == (2, 6)
    assert out.read_content.shape == (2, 16)
    assert out.read_position.shape == (2, 16)
    assert out.top_indices.shape == (2, 3)
    assert len(out.trace.selected_slot_ids) == 2
    assert "paamax_metadata" in out.trace.to_dict()
    assert torch.allclose(out.activation.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_geometry_aware_addressing_context_bias_vector_and_matrix():
    bank = make_bank()
    addressing = GeometryAwareAddressing(
        GeometryAwareAddressingConfig(dim=16, num_slots=6),
        slot_bank=bank,
    )
    query = torch.randn(2, 16)

    out_vec = addressing(query, context_geometry_bias=torch.zeros(6))
    out_mat = addressing(query, context_geometry_bias=torch.zeros(2, 6))

    assert out_vec.activation.shape == (2, 6)
    assert out_mat.activation.shape == (2, 6)


def test_bounded_associative_spread_normalization_and_bounds():
    spread = BoundedAssociativeSpread(
        BoundedAssociativeSpreadConfig(num_slots=6, max_steps=3, sparsity_top_k=3, spectral_norm_limit=1.25)
    )
    activation = torch.softmax(torch.randn(2, 6), dim=-1)
    out, trace = spread(activation, requested_steps=99)

    assert out.shape == (2, 6)
    assert trace.steps_requested == 99
    assert trace.steps_executed == 3
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
    report = spread.validate_transition()
    assert report["finite"] is True
    assert report["non_negative"] is True
    assert report["row_stochastic"] is True


def test_bounded_associative_spread_hebbian_update_is_safe():
    spread = BoundedAssociativeSpread(
        BoundedAssociativeSpreadConfig(num_slots=6, max_steps=2, sparsity_top_k=3, spectral_norm_limit=1.25)
    )
    activation = torch.softmax(torch.randn(4, 6), dim=-1)
    trace = spread.hebbian_update(activation)
    report = spread.validate_transition()

    assert trace.steps_executed == 0
    assert report["ok"] is True


def test_no_nan_inf_from_addressing_and_spread_with_extreme_inputs():
    bank = make_bank()
    addressing = GeometryAwareAddressing(
        GeometryAwareAddressingConfig(dim=16, num_slots=6),
        slot_bank=bank,
    )
    spread = BoundedAssociativeSpread(BoundedAssociativeSpreadConfig(num_slots=6))

    query = torch.randn(2, 16) * 1000.0
    out = addressing(query)
    spread_out, _ = spread(out.activation)

    assert torch.isfinite(out.scores).all()
    assert torch.isfinite(out.activation).all()
    assert torch.isfinite(spread_out).all()


def test_curved_resonant_core_accepts_wm1d_addressing_and_spread():
    bank = make_bank(dim=32, slots=8)
    addressing = GeometryAwareAddressing(
        GeometryAwareAddressingConfig(dim=32, num_slots=8),
        slot_bank=bank,
    )
    spread = BoundedAssociativeSpread(BoundedAssociativeSpreadConfig(num_slots=8))
    core = CurvedResonantWMCore(
        CurvedResonanceConfig(input_dim=32, hidden_dim=64, resonance_slots=8),
        geometry_aware_addressing=addressing,
        bounded_spread=spread,
    )

    x = torch.randn(2, 5, 32)
    y, trace = core(x, return_trace=True)
    assert y.shape == x.shape
    assert "geometry_aware_addressing" in trace["paamax_metadata"]
    assert "bounded_associative_spread" in trace["paamax_metadata"]
