from mnemonic_cortex.reasoning_depth import DepthIndexedSlotLattice, DepthLatticeConfig

def test_reason1a_capacity_metrics_multiplier_is_eight():
    lattice=DepthIndexedSlotLattice(DepthLatticeConfig(slot_count=64,key_dim=32,value_dim=32)); metrics=lattice.capacity_metrics()
    assert metrics['raw_slots']==64; assert metrics['depth_layers']==8; assert metrics['effective_subslots']==512; assert metrics['theoretical_capacity_multiplier']==8; assert len(metrics['active_depth_usage_histogram'])==8
