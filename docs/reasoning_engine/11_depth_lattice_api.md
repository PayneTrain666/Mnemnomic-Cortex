# REASON-1A Depth Lattice API

Primary classes: `DepthLatticeConfig`, `DepthIndexedSlotLattice`, `DepthWritePolicy`, `DepthLatticeTrace`, `SharedDepthSlotRegistry`.

```python
lattice = DepthIndexedSlotLattice(DepthLatticeConfig(slot_count=128, key_dim=256, value_dim=256))
value, attention, trace = lattice.read(query, return_trace=True)
proposal = lattice.propose_write(slot_index=0, value=value_tensor)
denied = lattice.commit_write(proposal, value=value_tensor)
allowed = lattice.commit_write(proposal, value=value_tensor, allow_mutation=True, write_permission=True)
metrics = lattice.capacity_metrics()
```

Writes are denied by default and return structured proposal results unless both explicit mutation flags are provided.
