# REASON-1D LTM Depth Integration API

```python
from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig

adapter = LTMDepthAdapter(LTMDepthAdapterConfig.disabled(key_dim=256))
same, trace = adapter.read_ltm(query, bank_name='cgmn_semantic', return_trace=True)

adapter = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=256, value_dim=256, slot_count=2048))
summary, trace = adapter.read_ltm(query, bank_name='hg_episodic', return_trace=True)

proposal = adapter.propose_consolidation(
    canonical_slot_id='concept.x',
    content='content',
    bank_name='cgmn_semantic',
    slot_index=1,
    depth_index=1,
    key=k,
    value=v,
)
```

Consolidation proposals are shadow-only by default. Permanent commit is deferred to an explicit downstream gate.
