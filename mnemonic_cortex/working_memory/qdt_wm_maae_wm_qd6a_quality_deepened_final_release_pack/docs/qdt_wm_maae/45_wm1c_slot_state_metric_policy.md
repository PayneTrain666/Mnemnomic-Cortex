# WM-1C CurvedSlotState and CurvatureMetricPolicy

## Source files created/updated

```text
mnemonic_cortex/working_memory/curved_slot_state.py
mnemonic_cortex/working_memory/curvature_metric_policy.py
mnemonic_cortex/working_memory/curved_resonant_wm_core.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm1c_slot_state_policy.py
```

## Implemented behavior

### CurvedSlotStateBank

Each slot carries:
- slot_id
- content
- position
- tangent
- phase
- curvature
- importance
- confidence
- last_updated
- trace_links

Includes:
- stable tensor views
- position radius clamp
- curvature clamp
- importance/confidence clamp
- update trace
- in-place repair
- snapshot API

### CurvatureMetricPolicy

Implements:
- global curvature
- per-slot curvature
- per-depth curvature
- context-conditioned curvature
- curvature clamps
- drift penalty
- PAAMA-X trace metadata
- [B,Z,S] combined curvature output

## Integration note

WM-1C provides standalone slot state and metric policy modules. WM-1D will integrate these directly into geometry-aware addressing and bounded associative spread.
