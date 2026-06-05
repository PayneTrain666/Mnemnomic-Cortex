# WM-1A Canonical EnhancedCurvedMemory Wrapper and Preservation

## Source files created/updated

```text
mnemonic_cortex/working_memory/legacy_enhanced_curved_memory.py
mnemonic_cortex/working_memory/wm_curved_core.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm1a_curved_preservation.py
```

## Preservation scope

The wrapper preserves the documented EnhancedCurvedMemory behavior:

- encoder
- curvature parameters
- memory slots
- memory importance
- associative weights
- content-based addressing
- associative activation spread
- read decode path
- write/update path
- process operation
- energy mode hook

## Canonical source location result

The project document contains the canonical wiring and behavioral description, but the mounted runtime does not contain a separate live repository file for `memory_curved.py`.

WM-1A therefore implements a canonical-compatible `EnhancedCurvedMemory` and keeps the wrapper able to delegate to a supplied external/canonical module when available.
