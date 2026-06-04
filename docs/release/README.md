# Release Documentation Index

Human-readable release documentation for Mnemonic Cortex. Machine-readable manifests,
pytest outputs, and ship checks live under [`release/`](../../release/).

## HGM / HPME (Hypergraph Manifold)

**Current baseline:** HGM v0.1 (HGM-0A through HGM-10)

| Doc | Purpose |
|---|---|
| [`docs/hgm_hpme/README.md`](../hgm_hpme/README.md) | Stage-by-stage HGM doc index |
| [`docs/hgm_hpme/HGM_V0_1_RELEASE_NOTES.md`](../hgm_hpme/HGM_V0_1_RELEASE_NOTES.md) | v0.1 release summary |
| [`docs/hgm_hpme/HGM_V0_1_API_FREEZE.md`](../hgm_hpme/HGM_V0_1_API_FREEZE.md) | Frozen public API surface |
| [`docs/hgm_hpme/HGM_V0_1_INTEGRATION_ROADMAP.md`](../hgm_hpme/HGM_V0_1_INTEGRATION_ROADMAP.md) | Post-v0.1 integration roadmap |
| [`release/HGM_V0_1_FINALIZATION_RECORD.md`](../../release/HGM_V0_1_FINALIZATION_RECORD.md) | Finalization record and test summary |

Implementation package: `mnemonic_cortex/hypergraph_manifold/` (56 modules, 176 API symbols at v0.1).

**HGM-QDT-AUDIT-1** (read-only bridge audit): [`docs/hgm_hpme/HGM_QDT_AUDIT_1_*`](../hgm_hpme/HGM_QDT_AUDIT_1_READ_ONLY_CONTRACT_AUDIT.md), [`release/hgm_qdt_audit_1/`](../../release/hgm_qdt_audit_1/). Verdict: bridge planning compatible; write-stage blocked. Tests: 153 HGM + 22 QDT/WM passed.

## Mnemonic Reasoning Engine

**Current closure:** REASON-4C (persistence line closure)

| Location | Contents |
|---|---|
| [`docs/reasoning_engine/`](../reasoning_engine/) | Design, API, shipcheck, and command docs per REASON stage |
| [`release/reason*_manifest.json`](../../release/) | Stage manifests (REASON-1A … REASON-4C) |

Implementation package: `mnemonic_cortex/reasoning_depth/`.

Backend authorization and real-backend hold docs: `docs/reasoning_engine/110` … `126`.

## QSPIN Bridge (QD6A)

**Current hold:** QSPIN-PROD-8-QD6A (`SHIP_FINAL_PRE_ACTIVATION_HOLD`)

| Location | Contents |
|---|---|
| Root `PROD5_*` … `PROD8_*` | PROD-5 … PROD-8 specs, ship checks, test results |
| [`docs/PROD7_*`](../PROD7_README.md), [`docs/PROD8_*`](../PROD8_README.md) | PROD-7/8 audit and readiness docs |
| [`docs/qdt_wm_maae/qspin_prod*`](../qdt_wm_maae/) | Per-stage QSPIN PROD design and shipcheck docs |
| [`release/qspin_prod*_qd6a_release_manifest.json`](../../release/) | PROD-4 … PROD-8 manifests |

Implementation modules: `mnemonic_cortex/working_memory/qspin_*.py`.

## WM quality and context compression

| Location | Contents |
|---|---|
| [`docs/qdt_wm_maae_quality/`](../qdt_wm_maae_quality/) | WM-QD quality deepening campaign |
| [`docs/qdt_wm_maae/110_context_compression_memory.md`](../qdt_wm_maae/110_context_compression_memory.md) | WM-CONTEXT-COMP-1A design |
| [`release/context_compression_memory_manifest.json`](../../release/context_compression_memory_manifest.json) | Context compression release manifest |
| [`release/wm_qd6a_release_manifest.json`](../../release/wm_qd6a_release_manifest.json) | WM-QD-6A baseline manifest |

## Codex / agent context

For QSPIN-BRIDGE agent bootstrap and stage manifests, see [`docs/chat_context/`](../chat_context/).

## Safety posture (cross-track)

All tracks above remain **additive and disabled-by-default** unless a future stage explicitly
authorizes activation:

- No live QDT/WM writes (HGM v0.1, REASON-4C, QSPIN PROD-8 hold).
- No production activation or commit execution.
- No real persistence backend writes without explicit authorization.
- Write-capable runtime integration requires a later explicit permission stage.
