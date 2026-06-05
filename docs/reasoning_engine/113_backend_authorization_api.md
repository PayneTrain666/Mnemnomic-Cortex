# Backend Authorization API

```python
from mnemonic_cortex.reasoning_depth import BackendAuthorizationGate, BackendAuthorizationConfig, BackendStoreTarget
gate = BackendAuthorizationGate(BackendAuthorizationConfig.planning_authorized(BackendStoreTarget.SQLITE))
decision = gate.decide()
payload = decision.to_dict()
```
