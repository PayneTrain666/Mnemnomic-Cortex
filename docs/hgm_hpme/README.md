# HGM / HPME Documentation

Hypergraph Manifold / Hyperset Probability Matrix Expansion (HGM/HPME) stage docs.

**Package:** `mnemonic_cortex/hypergraph_manifold/`  
**Current release:** HGM v0.1 (lineage HGM-0A through HGM-10)  
**Manifests:** [`release/hgm_*`](../../release/) and [`release/HGM_V0_1_*`](../../release/)

## Stage documentation

| Stage | Doc | Focus |
|---|---|---|
| HGM-0A | [`00_hgm_0a_foundation_types.md`](00_hgm_0a_foundation_types.md) | Foundation dataclasses, enums, validation, traces |
| HGM-0B | [`01_hgm_0b_probability_expander.md`](01_hgm_0b_probability_expander.md) | Probability expander, normalization, top-k extraction |
| HGM-1 | [`02_hgm_1_hyperedge_binder.md`](02_hgm_1_hyperedge_binder.md) | Hyperedge binding, coherence, conflict/opportunity graphs |
| HGM-2 | [`03_hgm_2_manifold_router.md`](03_hgm_2_manifold_router.md) | Manifold routing, geometry distance, depth retrieval |
| HGM-3 | [`04_hgm_3_spcp_procedural_memory.md`](04_hgm_3_spcp_procedural_memory.md) | SPCP procedural memory, action sequences, advisory robotics |
| HGM-4 | [`05_hgm_4_qdt_wm_bridge.md`](05_hgm_4_qdt_wm_bridge.md) | Dry-run QDT/WM bridge payloads, slot-lattice hooks |
| HGM-5 | [`06_hgm_5_embedding_evaluation.md`](06_hgm_5_embedding_evaluation.md) | Embedding scaffold, bridge evaluation, integration scoring |
| HGM-6 | [`07_hgm_6_write_permission_gate.md`](07_hgm_6_write_permission_gate.md) | Write-permission gate, transaction preview, rollback plan |
| HGM-7 | [`08_hgm_7_write_execution_adapter.md`](08_hgm_7_write_execution_adapter.md) | Simulation-mode write adapter, transaction log, recovery |
| HGM-8 | [`09_hgm_8_pipeline_benchmark.md`](09_hgm_8_pipeline_benchmark.md) | Runtime embedding, safe write replay, pipeline benchmark |
| HGM-9 | [`10_hgm_9_runtime_integration_readiness.md`](10_hgm_9_runtime_integration_readiness.md) | Runtime integration evaluation, production-readiness gate |
| HGM-10 | [`11_hgm_10_release_consolidation.md`](11_hgm_10_release_consolidation.md) | API freeze, release consolidation, documentation pack |

## HGM v0.1 release pack

| Doc | Purpose |
|---|---|
| [`HGM_V0_1_RELEASE_NOTES.md`](HGM_V0_1_RELEASE_NOTES.md) | Release summary and stage highlights |
| [`HGM_V0_1_API_FREEZE.md`](HGM_V0_1_API_FREEZE.md) | Frozen public API (176 symbols) |
| [`HGM_V0_1_INTEGRATION_ROADMAP.md`](HGM_V0_1_INTEGRATION_ROADMAP.md) | Post-v0.1 integration roadmap (6 items) |

Also see [`release/HGM_V0_1_FINALIZATION_RECORD.md`](../../release/HGM_V0_1_FINALIZATION_RECORD.md).

## Test chain (HGM-10 compatibility)

Per `release/HGM_V0_1_FINALIZATION_RECORD.json`:

- HGM-10 targeted: **10 passed**
- HGM-0A through HGM-10 compatibility: **153 passed**
- compileall: **passed**

Run the full chain:

```bash
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py tests/test_hgm_1_hyperedge_binder.py tests/test_hgm_2_manifold_router.py tests/test_hgm_3_spcp_procedural_memory.py tests/test_hgm_4_qdt_wm_bridge.py tests/test_hgm_5_embedding_evaluation.py tests/test_hgm_6_write_permission_gate.py tests/test_hgm_7_write_execution_adapter.py tests/test_hgm_8_pipeline_benchmark.py tests/test_hgm_9_runtime_integration_readiness.py tests/test_hgm_10_release_consolidation.py -q
```

## Safety posture

HGM v0.1 is **additive and evaluation-first**:

- No live QDT/WM writes.
- No mutation of working_memory/QDT internals.
- No production execution or robotics actuator calls.
- Write-capable runtime integration requires a later explicit permission stage.

## Next recommended command

```text
DEV-FLOW RUN HGM-QDT-AUDIT-1 — Read-Only QDT/WM Contract Audit, Bridge Compatibility Review, and Write-Stage Risk Register
```
