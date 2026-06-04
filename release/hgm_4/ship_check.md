# HGM-4 Ship Check

## Scope
Additive HGM/HPME bridge planning layer under `mnemonic_cortex/hypergraph_manifold/`.

## Commands

```bash
python -m pytest tests/test_hgm_4_qdt_wm_bridge.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py tests/test_hgm_2_manifold_router.py tests/test_hgm_3_spcp_procedural_memory.py tests/test_hgm_4_qdt_wm_bridge.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Results

- HGM-4 targeted tests: 18 passed
- HGM-0A through HGM-4 compatibility tests: 81 passed
- compileall: passed

## Safety

- Dry-run/read-only by default.
- No QDT/WM mutation.
- No live memory writes.
- No hardware calls.
- No network calls.
- Write-intent is blocked unless explicitly allowed for preview only.

## Files

See `changed_files.json`.
