# REASON-3A API

```python
from mnemonic_cortex.reasoning_depth import (
    ReasoningStrategyGraph,
    ReasoningStrategyGraphConfig,
    MultiPassThoughtPlanner,
    MultiPassThoughtPlannerConfig,
)

graph = ReasoningStrategyGraph(ReasoningStrategyGraphConfig.enabled_default())
graph.build_default_from_content(content='evidence context')
planner = MultiPassThoughtPlanner(MultiPassThoughtPlannerConfig.enabled_default())
report = planner.plan(query, content='evidence context')
payload = report.to_dict()
```
