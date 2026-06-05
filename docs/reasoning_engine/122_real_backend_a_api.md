# REAL-BACKEND-IMPLEMENTATION-A API

```python
from mnemonic_cortex.reasoning_depth import create_default_dry_run_backend, BackendPayloadEnvelope
backend = create_default_dry_run_backend()
envelope = BackendPayloadEnvelope(payload_kind='trace', payload={'ok': True}, idempotency_key='trace-1')
result = backend.dry_run_write(envelope).to_dict()
```
