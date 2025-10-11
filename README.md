# Enhanced Mnemonic Cortex — Fixed Codebase

This package provides a cleaned, runnable implementation of the **Mnemonic Cortex** with a **Triple Hybrid Memory** stack:
- **EnhancedHyperGeometricMemory** (fractal addressing + quantum-inspired phase modulation)
- **EnhancedCGMNMemory** (curvature-aware attention + safe slot writes)
- **EnhancedCurvedMemory** (lightweight associative working memory)

Key upgrades:
- All shapes consistent; no invalid `einsum` subscripts.
- Safe writes via `index_add_` accumulations + EMA.
- Fast pairwise L2 via GEMM trick (no `torch.cdist`).
- Lightbulb detector triggers *explosive recall* (temperature modulation across memories).
- Minimal ODE dynamics without extra deps.
- Tiny smoke-run and micro training loop included.

## Quickstart

```bash
# Python 3.9+ with PyTorch
python -m mnemonic_cortex.train_smoke
```

## Layout

- `mnemonic_cortex/utils.py` — seeds, fast L2, perf toggles
- `mnemonic_cortex/lightbulb.py` — LightbulbDetector + ExplosiveRecallScaler
- `mnemonic_cortex/memory_hg.py` — EnhancedHyperGeometricMemory
- `mnemonic_cortex/memory_cgmn.py` — EnhancedCGMNMemory
- `mnemonic_cortex/memory_curved.py` — EnhancedCurvedMemory
- `mnemonic_cortex/sensory_buffer.py` — EnhancedSensoryBuffer
- `mnemonic_cortex/triple_hybrid.py` — Hybrid wrapper
- `mnemonic_cortex/cortex.py` — EnhancedMnemonicCortex
- `mnemonic_cortex/train_smoke.py` — smoke test + tiny train loop
```