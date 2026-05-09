import torch

from mnemonic_cortex.working_memory import (
    CurvedLocalTraceBuilder,
    CurvedShadowWriteConfig,
    CurvedShadowWriteBuffer,
    CurvedSlotStateConfig,
    CurvedSlotStateBank,
    CurvedResonanceConfig,
    CurvedResonantWMCore,
)


def test_curved_local_trace_serializes_required_fields():
    resonance = {
        "confidence_score": 0.8,
        "novelty_score": 0.2,
        "paamax_metadata": {"trace_type": "curved_resonance"},
        "step_traces": [
            {"step": 0, "top_indices": [[1, 2]], "top_scores": [[0.9, 0.5]], "activation_entropy": [0.7], "delta_norm": 0.3}
        ],
        "inner_trace": {"top_indices": [[0, 1]]},
    }

    trace = (
        CurvedLocalTraceBuilder("read")
        .from_resonance_trace(resonance)
        .with_geometry_map("hierarchical")
        .with_curvature_state({"global": 0.1})
        .with_depth_contribution({"depth_0": 0.5})
        .with_disagreement(0.05)
        .build()
    )

    d = trace.to_dict()
    assert d["operation"] == "read"
    assert d["confidence"] == 0.8
    assert d["novelty"] == 0.2
    assert d["geometry_map"] == "hierarchical"
    assert d["curvature_state"]["global"] == 0.1
    assert d["depth_contribution"]["depth_0"] == 0.5
    assert d["disagreement"] == 0.05
    assert len(d["activation_route"]) == 1
    assert "events" in d


def test_shadow_write_stage_reject_missing_permission():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=4, dim=8))
    buffer = CurvedShadowWriteBuffer(
        CurvedShadowWriteConfig(dim=8, require_paamax_permission=True),
        slot_bank=bank,
    )

    proposal = buffer.stage(
        slot_indices=torch.tensor([0]),
        content_delta=torch.randn(1, 8),
        paamax_permission=False,
        confidence=1.0,
    )
    decision = buffer.evaluate(proposal.proposal_id)
    assert decision.decision == "reject"
    assert decision.reason == "paamax_permission_missing"


def test_shadow_write_commit_updates_slot_bank_and_history():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=4, dim=8))
    buffer = CurvedShadowWriteBuffer(
        CurvedShadowWriteConfig(dim=8, require_paamax_permission=True, interference_threshold=1.0),
        slot_bank=bank,
    )
    before = bank.snapshot([1]).content.clone()

    proposal = buffer.stage(
        slot_indices=torch.tensor([1]),
        content_delta=torch.ones(1, 8) * 0.05,
        paamax_permission=True,
        confidence=1.0,
        metadata={"test": True},
    )
    trace = CurvedLocalTraceBuilder("write").build()
    decision = buffer.commit(proposal.proposal_id, trace=trace)

    after = bank.snapshot([1]).content
    assert decision.committed is True
    assert decision.decision == "committed"
    assert buffer.pending_count() == 0
    assert len(buffer.history) >= 2  # commit_ready + committed
    assert not torch.allclose(before, after)


def test_shadow_write_manual_reject_removes_pending():
    buffer = CurvedShadowWriteBuffer(CurvedShadowWriteConfig(dim=8, require_paamax_permission=False))
    proposal = buffer.stage(
        slot_indices=torch.tensor([0]),
        content_delta=torch.randn(1, 8),
        paamax_permission=False,
    )
    decision = buffer.reject(proposal.proposal_id, reason="unit_test_reject")
    assert decision.decision == "reject"
    assert decision.reason == "unit_test_reject"
    assert buffer.pending_count() == 0


def test_curved_resonant_core_write_uses_shadow_buffer():
    bank = CurvedSlotStateBank(CurvedSlotStateConfig(num_slots=4, dim=8))
    shadow = CurvedShadowWriteBuffer(
        CurvedShadowWriteConfig(dim=8, require_paamax_permission=True, interference_threshold=1.0),
        slot_bank=bank,
    )
    core = CurvedResonantWMCore(
        CurvedResonanceConfig(input_dim=8, hidden_dim=16, resonance_slots=4),
        shadow_write_buffer=shadow,
    )

    x = torch.randn(2, 3, 8)
    y, trace = core(x, operation="write", return_trace=True)

    assert y.shape == x.shape
    assert trace["operation"] == "write"
    assert trace["paamax_metadata"]["trace_type"] == "curved_resonance_shadow_write"
    assert "shadow_write" in trace["paamax_metadata"]
    assert "local_trace" in trace["paamax_metadata"]


def test_curved_resonant_core_read_includes_local_trace():
    core = CurvedResonantWMCore(
        CurvedResonanceConfig(input_dim=8, hidden_dim=16, resonance_slots=4),
    )
    x = torch.randn(2, 3, 8)
    y, trace = core(x, operation="read", return_trace=True)

    assert y.shape == x.shape
    assert "curved_local_trace" in trace["paamax_metadata"]
    assert trace["paamax_metadata"]["curved_local_trace"]["operation"] == "read"
