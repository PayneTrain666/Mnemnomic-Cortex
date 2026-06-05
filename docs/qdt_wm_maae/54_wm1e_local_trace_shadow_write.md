# WM-1E Curved Local Trace and Curved Shadow Writes

## Source files created/updated

```text
mnemonic_cortex/working_memory/curved_local_trace.py
mnemonic_cortex/working_memory/curved_shadow_write.py
mnemonic_cortex/working_memory/curved_resonant_wm_core.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm1e_local_trace_shadow_write.py
```

## CurvedLocalTrace

Captures:
- selected slots
- activation route
- curvature state
- geometry map
- depth contribution
- confidence
- novelty
- disagreement
- write decision
- proposal ID
- PAAMA-X metadata
- local trace events

## CurvedShadowWriteBuffer

Implements:
- proposal staging
- PAAMA-X permission check
- confidence threshold check
- interference scoring
- commit-ready decision
- commit to CurvedSlotStateBank
- reject path
- decision history
- serializable state

## CurvedResonantWMCore integration

- read/process traces now include `curved_local_trace` metadata.
- write path uses shadow-write buffer when supplied.
- fallback write delegation remains available when no shadow buffer is supplied.
