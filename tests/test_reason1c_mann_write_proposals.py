import json
import torch

from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig


def test_reason1c_mann_hop_write_proposals_are_shadow_only():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=10))
    result = adapter.propose_hop_memory(
        slot_index=3,
        key=torch.randn(16),
        value=torch.randn(16),
        canonical_slot_id="concept.mann.1",
        hop_id=1,
    )

    assert result["committed"] is False
    assert result["shadow_only"] is True
    assert result["depth_routes"] == {
        "reasoning_transform": 4,
        "hop_history": 5,
        "candidate_hypothesis": 6,
        "scratch_trace": 7,
    }
    assert set(result["proposals"].keys()) == {
        "reasoning_transform",
        "hop_history",
        "candidate_hypothesis",
        "scratch_trace",
    }
    assert result["paamax_metadata"]["write_permission_granted"] is False
    json.dumps(result)


def test_reason1c_mann_commit_requires_explicit_permission():
    adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=10))
    proposals = adapter.propose_hop_memory(slot_index=0, key=torch.ones(16), value=torch.ones(16), canonical_slot_id="concept.mann.2")
    proposal_dict = proposals["proposals"]["reasoning_transform"]

    # Rebuild from a real proposal via the bank to avoid relying on dict internals.
    proposal = adapter.bank.lattice.propose_write(
        slot_index=0,
        depth_index=4,
        key=torch.ones(16),
        value=torch.ones(16),
        canonical_slot_id="concept.mann.2",
    )
    before = adapter.bank.values.clone()
    denied = adapter.bank.commit_write(proposal, key=torch.ones(16), value=torch.ones(16))
    assert denied["committed"] is False
    assert torch.equal(adapter.bank.values, before)

    allowed = adapter.bank.commit_write(
        proposal,
        key=torch.ones(16),
        value=torch.ones(16),
        allow_mutation=True,
        write_permission=True,
    )
    assert allowed["committed"] is True
    assert not torch.equal(adapter.bank.values, before)
