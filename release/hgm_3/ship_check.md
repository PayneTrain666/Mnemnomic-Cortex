# HGM-3 Ship Check

## Scope

HGM-3 adds an additive SPCP procedural-memory adapter under `mnemonic_cortex/hypergraph_manifold/`.

## Safety

- dry-run safe
- additive only
- no working_memory/QDT mutation
- no hardware calls
- no actuator commands
- advisory robotics planning output only

## Commands Run

```bash
python -m pytest tests/test_hgm_3_spcp_procedural_memory.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py tests/test_hgm_2_manifold_router.py tests/test_hgm_3_spcp_procedural_memory.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Results

- HGM-3 targeted tests: 15 passed
- HGM-0A + HGM-0B + HGM-1 + HGM-2 + HGM-3 compatibility tests: 63 passed
- compileall: passed

## Known Limits

- deterministic dependency-light SPCP embeddings only
- no learned procedural embedding model yet
- no QDT/WM runtime bridge yet
- no live robotics control or actuator execution
- no embodied policy deployment

## No-Destructive-Change Check

Only `mnemonic_cortex/hypergraph_manifold/__init__.py` was modified among existing source files. New implementation files, docs, tests, and release metadata were added. No files were deleted.
