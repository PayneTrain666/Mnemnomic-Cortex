# REASON-4C Persistence Line Closure API

```python
from mnemonic_cortex.reasoning_depth import PersistenceLineClosure, PersistenceLineClosureConfig
closure = PersistenceLineClosure(PersistenceLineClosureConfig.enabled_default())
report = closure.close()
payload = report.to_dict()
```
