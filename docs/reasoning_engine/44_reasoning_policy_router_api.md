# REASON-2B Policy Router API

```python
from mnemonic_cortex.reasoning_depth import ReasoningController, ReasoningControllerConfig, ReasoningPolicyRouterConfig

cfg = ReasoningControllerConfig(
    enabled=True,
    key_dim=256,
    value_dim=256,
    slot_count=2048,
    use_policy_router=True,
    policy_router_config=ReasoningPolicyRouterConfig.enabled_default(task_mode='hypothesis'),
)
controller = ReasoningController(cfg)
result = controller.run_reasoning_pass(query, content='project hypothesis')
payload = result.to_dict()
```

Policy routing remains non-mutating. It affects route metadata, hop bounds, LTM bank choice, and consolidation confidence/disagreement inputs only.
