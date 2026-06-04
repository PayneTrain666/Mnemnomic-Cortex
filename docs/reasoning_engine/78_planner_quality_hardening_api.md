# REASON-3C Planner Quality Hardening API

```python
from mnemonic_cortex.reasoning_depth import ControllerPlannerIntegration, ControllerPlannerIntegrationConfig
integration = ControllerPlannerIntegration(ControllerPlannerIntegrationConfig.enabled_default())
report = integration.run(query, content='context')
payload = report.to_dict()
```
