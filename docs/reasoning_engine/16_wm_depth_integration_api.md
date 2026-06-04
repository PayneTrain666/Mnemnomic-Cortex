# REASON-1B WM Depth Integration API

```python
from mnemonic_cortex.reasoning_depth import WMDepthController

controller = WMDepthController.disabled(input_dim=256)
same, trace = controller.process_wm(x, return_trace=True)

controller = WMDepthController.enabled_default(input_dim=256, value_dim=256, slot_count=64)
summary, trace = controller.process_wm(x, return_trace=True)

result = controller.route_context_candidate(context=x, response=y, candidate=candidate)
```

Writes are proposal-only. Permanent mutation still requires the downstream explicit mutation/write-permission gate from REASON-1A.
