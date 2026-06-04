# HGM-8 Ship Check

## Commands

```bash
python -m pytest tests/test_hgm_8_pipeline_benchmark.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py tests/test_hgm_2_manifold_router.py tests/test_hgm_3_spcp_procedural_memory.py tests/test_hgm_4_qdt_wm_bridge.py tests/test_hgm_5_embedding_evaluation.py tests/test_hgm_6_write_permission_gate.py tests/test_hgm_7_write_execution_adapter.py tests/test_hgm_8_pipeline_benchmark.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Results

- HGM-8 targeted tests: 13 passed.
- HGM-0A through HGM-8 compatibility tests: 130 passed.
- compileall: passed.
- ZIP integrity: verified after packaging.

## Safety

HGM-8 is evaluation-first. It performs no live QDT/WM writes, no hardware calls,
no network calls, and no mutation of existing working-memory internals.
