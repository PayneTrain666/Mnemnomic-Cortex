import torch, pytest
from mnemonic_cortex.reasoning_depth import compute_slot_depth_attention, depth_entropy, DepthEntropyError

def test_reason1a_depth_entropy_finite_and_bounded():
    ent=depth_entropy(torch.ones(4,8)/8.0, dim=-1, normalise=True)
    assert ent.shape == (4,); assert torch.all(ent <= 1.00001); assert torch.all(ent >= 0.0)

def test_reason1a_depth_entropy_rejects_nan():
    probs=torch.ones(2,8)/8.0; probs[0,0]=float('nan')
    with pytest.raises(DepthEntropyError): depth_entropy(probs)

def test_reason1a_slot_depth_attention_shapes():
    result=compute_slot_depth_attention(torch.randn(2,16), torch.randn(20,8,16), read_top_k_slots=5, read_top_k_depths=3)
    assert result.slot_depth_scores.shape == (2,20,8); assert result.top_slot_indices.shape == (2,5); assert result.top_depth_indices.shape == (2,5,3); assert torch.isfinite(result.depth_entropy).all()
