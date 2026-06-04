# REASON-4B Persistence Backend API

```python
from mnemonic_cortex.reasoning_depth import PersistenceBackendStub, PersistenceBackendConfig
backend = PersistenceBackendStub(PersistenceBackendConfig.enabled_default())
result = backend.dry_run_write({'payload_id': 'p1', 'items': [{'id': 'x'}]})
payload = result.to_dict()
```
