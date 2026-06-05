# HGM-1 Ship Check

## Status
PASS

## Commands

```bash
python -m pytest tests/test_hgm_1_hyperedge_binder.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Results

- Targeted HGM-1 tests: 12 passed.
- HGM-0A + HGM-0B + HGM-1 compatibility tests: 33 passed.
- compileall: passed.
- Existing working_memory/QDT source files: not modified.

## Known limits

- HGM-1 is deterministic/heuristic. Learned hypergraph scoring comes later.
- Conflict/opportunity detection is conservative and metadata-light.
- HGM-2 must add manifold chart routing and geometry-aware distance wrappers.
