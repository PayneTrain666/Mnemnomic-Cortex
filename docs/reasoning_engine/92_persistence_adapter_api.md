# REASON-4A Persistence Adapter API

```python
from mnemonic_cortex.reasoning_depth import ReasoningPersistenceAdapter, ReasoningPersistenceAdapterConfig
adapter = ReasoningPersistenceAdapter(ReasoningPersistenceAdapterConfig.enabled_default())
report = adapter.prepare(target_store='reasoning_trace', item_kind='trace', items=[{'id': 't1'}])
payload = report.to_dict()
```
