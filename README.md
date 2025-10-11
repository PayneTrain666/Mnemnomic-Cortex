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

# Mnemonic Cortex

Neuroscience-inspired memory architecture written in PyTorch.

## Installation

```bash
git clone https://github.com/PayneTrain666/Mnemnomic-Cortex.git
cd Mnemnomic-Cortex
pip install -r requirements.txt
```

## Quick smoke-test

```bash
python -m mnemonic_cortex.train_smoke
# ➜ Smoke OK: torch.Size([8, 5, 128]) torch.Size([8, 128])
```

## Benchmarking

The repo ships with two synthetic sequence tasks:

* **CopyTask** – reproduce the input sequence.
* **RecallTask** – output the last symbol before a marker.

Baseline models: LSTM, tiny Transformer, and **EnhancedMnemonicCortex**.

Run all benchmarks (one epoch each, quick):

```bash
python -m benchmark.runner
# cortex/copy:   loss=… acc=…
# transformer/…
# … results.csv saved
```

## Architecture overview

```
sensory buffer  →  working memory (curved)  →  long-term memory (HG | CGMN | curved)
                                 ↘  importance-gated write  ↗
```

Key components:

| Module | Purpose |
|--------|---------|
| `EnhancedSensoryBuffer` | Filters raw input with GRU + self-attention. |
| `EnhancedCurvedMemory` | Acts as working memory with associative retrieval. |
| `EnhancedTripleHybridMemory` | Combines three specialised LTMs (hyper-geometric, CGMN, curved). |
| `LightbulbDetector` | Detects salient "aha" moments and adjusts temperature. |

## Temperature control

```python
ctx = torch.randn(B, d)
cortex = EnhancedMnemonicCortex(d,d)
cortex.set_temperature(torch.tensor(0.7))  # sharper attention
```

## Contributing & Tests

* Run `pytest` to execute shape/unit tests (coming soon).
* A GitHub Action checks smoke-test on every PR.

## Limitations & Future Work

> This project is research-grade. The architecture intentionally pushes boundaries and, as a result, comes with known trade-offs. Below is a condensed list of the main caveats and the roadmap to address them.

### Where dragons lurk

• **Non-differentiable, in-place memory writes** – Fine for single-GPU research but unsafe under DistributedDataParallel (DDP) because replicas drift.  
• **Brute-force fractal search** – O(B S M D) distance calc hits latency when `mem_slots` is large.  
• **Hologram interference** – Spectral energy can blow up without stronger normalization.  
• **Noisy light-bulb triggering** – Simple z-score on a single projection; uncalibrated for real data.  
• **Fancy mechanics lack supervision** – Without explicit recall losses the model may under-utilise HG/CGMN.  
• **Checkpoint compatibility** – No versioned schema yet; buffer shapes could change.

### Actionable roadmap (excerpt)

Priority | Item
:-- | :--
P0 | Sync external memory writes across DDP; clamp & renorm holograms; add unit tests for fire-mask behaviour.
P1 | Approximate NN search (e.g., FAISS IVF-PQ); cache FFT keys; micro-batch hologram updates.
P2 | Contrastive recall loss; learned write gate with STE; adaptive light-bulb threshold.
P3 | Collision control & slot re-assignment; versioned checkpoints; richer metric logging.

See `docs/ROADMAP.md` for the full discussion.