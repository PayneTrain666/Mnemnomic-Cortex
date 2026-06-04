# REASON-2C Evidence / Counterfactual API

```python
from mnemonic_cortex.reasoning_depth import (
    ReasoningController,
    ReasoningControllerConfig,
    EvidenceReasoningConfig,
    CounterfactualProbeConfig,
    ConflictAwareConsolidationConfig,
)

cfg = ReasoningControllerConfig(
    enabled=True,
    key_dim=256,
    value_dim=256,
    slot_count=2048,
    use_evidence_reasoning=True,
    evidence_config=EvidenceReasoningConfig.enabled_default(),
    use_counterfactual_probe=True,
    counterfactual_config=CounterfactualProbeConfig.enabled_default(),
    use_conflict_aware_consolidation=True,
    conflict_config=ConflictAwareConsolidationConfig.enabled_default(),
)
controller = ReasoningController(cfg)
result = controller.run_reasoning_pass(query, content='Evidence one. Evidence two.')
```
