import json, torch
from mnemonic_cortex.reasoning_depth import DepthIndexedSlotLattice, DepthLatticeConfig, DepthRole, DepthWriteMode, DepthWritePolicy

def test_reason1a_write_policy_single_and_multi_proposals_are_serializable():
    cfg=DepthLatticeConfig(slot_count=8,key_dim=16,value_dim=16); policy=DepthWritePolicy(cfg); value=torch.randn(16); key=torch.randn(16)
    single=policy.single_depth(slot_index=2, depth_index=4, value=value, key=key, canonical_slot_id='concept.1')
    multi=policy.multi_depth(slot_index=2, depth_indices=[0,1,5], value=value)
    assert single.depth_indices == [4]; assert multi.depth_indices == [0,1,5]; json.dumps(single.to_dict()); json.dumps(multi.to_dict())

def test_reason1a_role_coded_write_maps_roles_to_depths():
    cfg=DepthLatticeConfig(slot_count=8,key_dim=16,value_dim=16); policy=DepthWritePolicy(cfg)
    proposal=policy.role_coded(slot_index=3, role_values={DepthRole.Z1_SEMANTIC_INVARIANT:torch.randn(16), DepthRole.Z5_TEMPORAL_EPISODE:torch.randn(16)}, canonical_slot_id='concept.2')
    assert proposal.mode == DepthWriteMode.ROLE_CODED; assert proposal.depth_indices == [1,5]

def test_reason1a_lattice_propose_write_shadow_only_default():
    lattice=DepthIndexedSlotLattice(DepthLatticeConfig(slot_count=8,key_dim=16,value_dim=16)); proposal=lattice.propose_write(slot_index=1, value=torch.randn(16)); assert proposal.depth_indices == [7]; assert proposal.to_dict()['metadata']['shadow_only_by_default'] is True
