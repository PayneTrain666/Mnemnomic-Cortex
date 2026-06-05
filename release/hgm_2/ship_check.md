# HGM-2 Ship Check

## Scope
Additive HGM/HPME package extension only. No working_memory, QDT, SPCP, or existing memory internals were modified.

## Files added/changed
See `changed_files.json`.

## Commands

```bash
python -m pytest tests/test_hgm_2_manifold_router.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py tests/test_hgm_2_manifold_router.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Results

- HGM-2 targeted tests: 15 passed.
- HGM-0A + HGM-0B + HGM-1 + HGM-2 compatibility tests: 48 passed.
- compileall: passed.

## Reliability / safety

- Fail-closed validation for invalid charts, unsupported geometry, invalid hyperedges, and invalid Poincare points.
- Structured empty results for empty hyperedge inputs.
- Missing coordinates degrade to explicit fallback assignments only when allowed.
- Trace records emitted for validation, creation, warning, and failure events.
- No hard torch/numpy dependency.

## Known limits

- HGM-2 is deterministic and heuristic.
- Learned manifold embeddings are deferred.
- Runtime QDT/WM bridge is deferred.
- SPCP procedural memory and robotics planning bridge are deferred to HGM-3.
