# WM-1D Geometry-Aware Addressing and Bounded Associative Spread

## Source files created/updated

```text
mnemonic_cortex/working_memory/geometry_aware_addressing.py
mnemonic_cortex/working_memory/bounded_associative_spread.py
mnemonic_cortex/working_memory/curved_resonant_wm_core.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm1d_geometry_addressing_spread.py
```

## GeometryAwareAddressing

Addressing score combines:
- content similarity
- curved/Poincare-style distance
- phase compatibility
- importance
- confidence
- trace reliability
- context geometry bias
- curvature policy adjustment

Outputs:
- activation [B,S]
- read_content [B,D]
- read_position [B,D]
- top indices/scores
- PAAMA-X trace metadata

## BoundedAssociativeSpread

Implements:
- non-negative transition matrix
- row-stochastic normalization
- optional top-k sparsity
- spectral norm clamp
- bounded spread steps
- decay
- entropy floor
- bounded Hebbian update

## CurvedResonantWMCore integration

The resonant core now accepts optional:
- GeometryAwareAddressing
- BoundedAssociativeSpread

It uses them to enrich resonance seed and PAAMA-X metadata without replacing the preserved inner curved core.
