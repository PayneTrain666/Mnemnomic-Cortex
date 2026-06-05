# WM-1B Curved Resonant WM Core Upgrade

## Source files created/updated

```text
mnemonic_cortex/working_memory/curved_resonant_wm_core.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm1b_curved_resonant_core.py
```

## Implemented behavior

- Preserves WMCurvedAssociativeCore as the inner core.
- Adds bounded resonance/lightbulb loop.
- Adds local novelty/resonance scoring.
- Adds bounded resonance iterations.
- Adds trace events for activation path and resonance.
- Adds PAAMA-X-compatible trace metadata.
- Delegates writes to preserved inner core pending WM-1E curved shadow writes.

## Out of scope

- CurvedSlotState and CurvatureMetricPolicy are WM-1C.
- Geometry-aware addressing and bounded associative spread are WM-1D.
- Curved shadow writes are WM-1E.
