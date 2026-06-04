# Release Manifest Index

This folder holds machine-readable release manifests, ship checks, pytest outputs, and
finalization records for Mnemonic Cortex release tracks. Each manifest is the source of
truth for its stage scope, safety posture, and test results.

## Active release tracks

| Track | Stages | Manifest(s) | Status |
|---|---|---|---|
| **HGM / HPME** | HGM-0A … HGM-10, HGM v0.1 | `hgm_*/manifest.json`, `HGM_V0_1_*` | SHIP — additive, evaluation-first |
| **Mnemonic Reasoning** | REASON-1A … REASON-4C | `reason*_manifest.json` | SHIP — disabled-by-default, no real writes |
| **Backend authorization** | FUTURE-BACKEND-AUTH, REAL-BACKEND-A | `future_backend_*`, `real_backend_*` | HOLD — dry-run only |
| **QSPIN bridge (QD6A)** | PROD-4 … PROD-8 | `qspin_prod*_qd6a_release_manifest.json` | PROD-8 on final pre-activation hold |
| **WM context compression** | WM-CONTEXT-COMP-1A | `context_compression_memory_manifest.json` | SHIP — 6 targeted tests passed |
| **WM QD6A quality** | WM-QD-6A | `wm_qd6a_release_manifest.json` | Historical baseline reference |

## HGM / HPME lineage (HGM-0A through HGM-10)

Package path: `mnemonic_cortex/hypergraph_manifold/`

| Stage | Title | Targeted tests | Compatibility chain |
|---|---|---|---|
| HGM-0A | Foundation types | see `hgm_0a/` | baseline |
| HGM-0B | Probability expander, normalization, top-k | 13 passed | 21 passed |
| HGM-1 | Hyperedge binder, coherence, conflict/opportunity | 12 passed | 33 passed |
| HGM-2 | Manifold router, geometry distance, depth retrieval | see `hgm_2/` | see manifest |
| HGM-3 | SPCP procedural memory, robotics planning bridge | 15 passed | 63 passed |
| HGM-4 | QDT/WM bridge, slot-lattice hooks, trace-safe integration | 18 passed | 81 passed |
| HGM-5 | Embedding trainer scaffold, bridge evaluation | 13 passed | 94 passed |
| HGM-6 | Write-permission gate, transaction preview, rollback plan | 12 passed | 106 passed |
| HGM-7 | Write execution adapter (simulation), transaction log | 11 passed | 117 passed |
| HGM-8 | Runtime embedding, safe write replay, pipeline benchmark | 13 passed | 130 passed |
| HGM-9 | Runtime integration evaluation, production-readiness gate | 13 passed | 143 passed |
| HGM-10 | API freeze, release consolidation, roadmap | 10 passed | 153 passed |

**HGM v0.1 finalization:** see `HGM_V0_1_FINALIZATION_RECORD.json`, `HGM_V0_1_RELEASE_NOTES.md`.

Per-stage docs: `docs/hgm_hpme/00_` … `11_` plus `HGM_V0_1_*`.

## HGM v0.1 safety boundaries (all stages)

- Additive only; no mutation of QDT/WM internals.
- No live QDT/WM writes; no production execution.
- No robotics actuator execution; advisory planning only through HGM-3.
- Write-capable runtime integration requires a later explicit permission stage.

## QSPIN PROD-4 … PROD-8 (QD6A)

| Stage | Ship decision | Pytest / manual summary |
|---|---|---|
| PROD-4 | synthetic harness + observability | 18 passed |
| PROD-5 | shadow runtime + canary | 26 passed |
| PROD-6 | stress replay + CI matrix | 17 pytest passed |
| PROD-7 | readonly runtime probe | 16 pytest passed |
| PROD-8 | `SHIP_FINAL_PRE_ACTIVATION_HOLD` | 10 pytest passed; `production_active=false` |

Safety boundaries (all PROD stages): commit execution, live routing, external/QH/shared-slot
writes, topology execution, and production activation remain **BLOCKED**.

Docs: root `PROD5_*` … `PROD8_*`, `docs/PROD7_*`, `docs/PROD8_*`, `docs/qdt_wm_maae/qspin_prod*`.

## Mnemonic Reasoning (REASON-1A … REASON-4C)

Package path: `mnemonic_cortex/reasoning_depth/`

- **REASON-1A–1D:** depth lattice, WM/MANN/LTM adapters.
- **REASON-2A–2D:** controller, policy router, evidence/counterfactual passes, regression matrix.
- **REASON-3A–3D:** multi-pass planner, API freeze, release candidate.
- **REASON-4A–4C:** commit interface, dry-run persistence backends, line closure.

REASON-4C compatibility: **106 passed**. Defaults inert; no real store writes.

Docs: `docs/reasoning_engine/`.

## Next recommended command (HGM v0.1)

```text
DEV-FLOW RUN HGM-QDT-AUDIT-1 — Read-Only QDT/WM Contract Audit, Bridge Compatibility Review, and Write-Stage Risk Register
```
