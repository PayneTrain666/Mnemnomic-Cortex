# REASON-3D Release Candidate API

```python
from mnemonic_cortex.reasoning_depth import ReasoningReleaseCandidate, ReasoningReleaseCandidateConfig
rc = ReasoningReleaseCandidate(ReasoningReleaseCandidateConfig.enabled_default())
report = rc.evaluate()
payload = report.to_dict()
```
