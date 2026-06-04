# REASON-1C MANN Depth Integration API

```python
from mnemonic_cortex.reasoning_depth import MANNDepthAdapter, MANNDepthAdapterConfig

adapter = MANNDepthAdapter(MANNDepthAdapterConfig.disabled(key_dim=256))
same, trace = adapter.read_hop(query, hop_id=0, return_trace=True)

adapter = MANNDepthAdapter(MANNDepthAdapterConfig.enabled_default(key_dim=256, value_dim=256, slot_count=512))
summary, trace = adapter.read_hop(query, hop_id=1, return_trace=True)

proposal = adapter.propose_hop_memory(slot_index=3, key=k, value=v, canonical_slot_id='concept.x', hop_id=1)
```

Write proposals are shadow-only by default. Explicit mutation still requires the lower-level commit gate flags.
