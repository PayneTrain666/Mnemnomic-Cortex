import torch
from mnemonic_cortex.reasoning_depth import DepthIndexedSlotLattice, DepthLatticeConfig, DepthWriteMode

def test_reason1a_commit_write_requires_explicit_mutation_and_permission():
    lattice=DepthIndexedSlotLattice(DepthLatticeConfig(slot_count=8,key_dim=16,value_dim=16)); before=lattice.values.clone()
    proposal=lattice.propose_write(slot_index=0, depth_index=1, value=torch.ones(16), key=torch.ones(16), mode=DepthWriteMode.SINGLE, canonical_slot_id='concept.no_mutation')
    denied=lattice.commit_write(proposal, value=torch.ones(16), key=torch.ones(16)); assert denied['committed'] is False; assert torch.equal(lattice.values, before)
    allowed=lattice.commit_write(proposal, value=torch.ones(16), key=torch.ones(16), allow_mutation=True, write_permission=True); assert allowed['committed'] is True; assert not torch.equal(lattice.values, before); assert lattice.canonical_slot_ids[0] == 'concept.no_mutation'
