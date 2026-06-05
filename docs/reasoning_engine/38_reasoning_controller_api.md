# REASON-2A Reasoning Controller API

```python
from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig

controller = ReasoningController(ReasoningControllerConfig.disabled(key_dim=256))
result = controller.run_reasoning_pass(query, content='input')

controller = ReasoningController(ReasoningControllerConfig.enabled_default(key_dim=256, value_dim=256, slot_count=2048))
result = controller.run_reasoning_pass(query, content='reasoning content', project_id='project', chat_id='chat', episode_id='episode', write_permission=False)
payload = result.to_dict()
```

Default mode is disabled/pass-through. Enabled mode remains non-mutating and shadow-only by default.
