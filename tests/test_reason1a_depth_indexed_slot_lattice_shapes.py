import json, torch
from mnemonic_cortex.reasoning_depth import DepthIndexedSlotLattice, DepthLatticeConfig, DepthReadMode

def test_reason1a_lattice_shapes_and_read_trace():
    cfg=DepthLatticeConfig(slot_count=12,key_dim=32,value_dim=40,read_top_k_slots=4,read_top_k_depths=2)
    lattice=DepthIndexedSlotLattice(cfg)
    assert lattice.keys.shape == (12,8,32); assert lattice.values.shape == (12,8,40); assert lattice.importance.shape == (12,8); assert lattice.confidence.shape == (12,8)
    value, attention, trace = lattice.read(torch.randn(3,32), read_mode=DepthReadMode.TOP_K, return_trace=True)
    assert value.shape == (3,40); assert attention.slot_depth_scores.shape == (3,12,8); assert attention.slot_attention.shape == (3,12); assert attention.depth_attention.shape == (3,12,8); assert torch.isfinite(value).all(); json.dumps(trace); assert trace['paamax_metadata']['trace_governance'] is True

def test_reason1a_lattice_accepts_token_query_btd():
    cfg=DepthLatticeConfig(slot_count=10,key_dim=16,value_dim=16,read_top_k_slots=3,read_top_k_depths=2)
    lattice=DepthIndexedSlotLattice(cfg); value=lattice.read(torch.randn(2,5,16)); assert value.shape == (2,16); assert torch.isfinite(value).all()
