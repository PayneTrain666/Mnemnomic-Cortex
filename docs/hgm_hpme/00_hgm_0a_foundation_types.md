# HGM-0A — Hypergraph Manifold + Hyperset Probability Matrix Foundation Types

## Purpose

HGM-0A creates the first concrete implementation layer for the Mnemonic Cortex
**Hypergraph Manifold / Hyperset Probability Matrix Expansion** extension.

The release is deliberately additive:

- New package path: `mnemonic_cortex/hypergraph_manifold/`
- New tests: `tests/test_hgm_0a_foundation_types.py`
- No existing QDT/WM files are modified.

## Foundation Objects

- `MutationToken`
- `HypersetMatrix`
- `ScenarioHyperedge`
- `ManifoldChart`
- `DepthLayerAssignment`
- `QSpinSignature`
- `TraceRecord`
- `ValidationResult`

## Tensor Shape Contracts

| Contract | Axes | Meaning |
|---|---|---|
| `P[v,m]` | variable, mutation | basic probability matrix |
| `P[v,m,d]` | variable, mutation, depth | depth-aware tensor |
| `P[v,m,d,c]` | variable, mutation, depth, context | context-aware tensor |
| `P[v,m,d,c,t]` | variable, mutation, depth, context, time | temporal tensor |
| `P[v,m,d,c,t,a]` | variable, mutation, depth, context, time, action | planning tensor |

## Validation Rules

- Probabilities must be finite.
- Probabilities must be non-negative.
- Mutation-axis probability groups must sum to 1.0 unless normalization is disabled.
- Depth layers are bounded to canonical D0-D7.
- Geometry assignments must be from the HGM geometry enum.
- Hyperedges require at least two unique nodes unless singleton mode is explicitly enabled.
- Q-spin signatures must match configured dimensionality.
- Trace records must have stable IDs, component names, timestamps, and lineage.
- Trace payload redaction blocks obvious secret-bearing keys.

## Reliability / Stability / Security Layer

- Fail-closed coercion for enums.
- Explicit validation objects rather than silent mutation.
- No destructive branch changes.
- No dependency on torch/numpy for foundation types.
- No external I/O in validation paths.
- Redaction helper for obvious secret-bearing trace payload keys.
- Tests cover valid and invalid construction paths.

## Lineage

`MnemonicCortex → HGM → HPME → HGM-0A`
