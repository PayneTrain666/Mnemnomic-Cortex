# Enhanced Mnemonic Cortex — Quantum Holographic Edition (v3.1)

This package provides a runnable implementation of the **Mnemonic Cortex** with a **Triple Hybrid Memory** stack:
- **EnhancedHyperGeometricMemory**: fractal addressing + **complex holographic storage** (cHRR via FFT) + usage tracking / energy mode + **lightbulb/explosive recall**
- **EnhancedCGMNMemory**: curvature-aware attention + safe slot writes + usage consolidation + **lightbulb/explosive recall**
- **EnhancedCurvedMemory**: lightweight associative working memory with usage/energy controls

## What’s new in v3.1
- Correct **Ricci-flow einsum** in HG: `z = einsum('bseq,ed->bsdq', z, R)`
- **Explosive recall** (read & write) for HG + CGMN + LTM:
  - Lowers effective temperature (sharper search)
  - Boosts `top-k` on fire
  - Increases write plasticity (HG hologram LR up; CGMN EMA down)
- LTM reads now receive the **fire mask** via the cortex, not just writes.

## Quickstart

```bash
python -m mnemonic_cortex.train_smoke
