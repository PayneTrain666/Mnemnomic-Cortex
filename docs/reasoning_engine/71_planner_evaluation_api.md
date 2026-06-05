# REASON-3B Planner Evaluation API

```python
from mnemonic_cortex.reasoning_depth import PlannerEvaluator, PlannerEvaluationConfig
evaluator = PlannerEvaluator(PlannerEvaluationConfig.enabled_default())
evaluation = evaluator.evaluate(thought_plan_report)
payload = evaluation.to_dict()
```
