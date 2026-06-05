# REASON-2D Reasoning Controller API

```python
from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig

cfg = ReasoningControllerAPIConfig(
    enabled=True,
    key_dim=256,
    value_dim=256,
    slot_count=2048,
    allow_policy_router=True,
    allow_evidence_reasoning=True,
)
api = ReasoningControllerAPI(cfg)
result = api.run_reasoning_pass(query, content='project evidence')
payload = result.to_dict()
```
