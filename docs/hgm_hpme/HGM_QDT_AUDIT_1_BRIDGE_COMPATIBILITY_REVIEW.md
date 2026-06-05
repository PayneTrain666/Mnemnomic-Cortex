# HGM-QDT-AUDIT-1 — Bridge Compatibility Review

## Verdict

HGM v0.1 is **compatible for read-only bridge planning**, **compatible for dry-run replay/evaluation**, and **blocked for real write execution** until a dedicated write-preparation stage resolves the contract gaps.

## Compatible surfaces

| Surface | HGM side | QDT/WM side | Status |
|---|---|---|---|
| Adapter detection | `detect_qdt_wm_adapter_status` | `mnemonic_cortex.working_memory` package | Compatible for presence detection |
| Bridge payloads | `HGMBridgePayload` | Potential input source for write proposal adapter | Partial; summary-only today |
| Slot hooks | `SharedSlotLatticeHook` | `SharedSlotRegistry` / `SharedSlotStore` | Partial; identity mapping required |
| Write permission | `WritePermissionState` / `HGM6CommitOptions` | `SystemWriteProposal.write_permission` / `SystemCommitGate.require_write_permission` | Partial; two-key handshake required |
| Transaction preview | `TransactionCommitPreview` | `SystemCommitGate.stage/evaluate/commit` | Partial; preview only today |
| Rollback | `RollbackManifest` / `RecoveryVerificationResult` | `SystemCommitGate.rollback_stack` / `rollback_last` | Partial; snapshot binding required |
| QH / q-spin | `qspin_placeholder_*` and HGM metadata | `QuantumHolographicStorage`, `QHCodeSchema`, `QHStorageRecord` | Blocked; real conversion contract required |

## QDT/WM surfaces identified

- `QDTWorkingMemory`
- `SystemWriteProposal`
- `SystemCommitGate`
- `CommitGateDecision`
- `CommitGateEvaluation`
- `SharedSlotStore`
- `SharedSlotRegistry`
- `SharedSlotRecord`
- `canonical_slot_id`
- `QuantumHolographicStorage`
- `QHCodeSchema`
- `QHStorageRecord`
- `WMTrace`, `TraceItem`, `WMTraceEmitter`

## Required write-stage mapping table

| Required mapping | Current HGM field | Required QDT/WM field | Status |
|---|---|---|---|
| Content vector | `HGMBridgePayload.content_summary`, embeddings, SPCP vectors | `SystemWriteProposal.content: torch.Tensor [D]` | Missing |
| Local slot | `SharedSlotLatticeHook.target_slot_id` | `SystemWriteProposal.local_slot_id` | Needs mapping |
| Canonical slot | generated after WM write | `SharedSlotWriteResult.canonical_id`, `css-*` | Missing lineage link |
| Depth | `DepthLayer` enum | `depth_index: int` | Needs explicit mapping |
| Geometry | `GeometryType` enum | `geometry_map: str` | Needs explicit mapping |
| Q-spin/QH | `qspin_signature_id` | `QHCodeSchema` / `QHStorageRecord` fields | Missing conversion |
| Permission | `WritePermissionState` | `write_permission` bool + gate config | Needs two-key handshake |
| Rollback | `RollbackManifest.previous_state_ref` | actual pre-commit snapshots | Missing binding |

## Compatibility review conclusion

The next stage must not jump directly to production writes. It should be a **write-preparation audit/adapter stage** that builds the missing proposal-materialization and contract-probe layer, still without mutating QDT/WM.
