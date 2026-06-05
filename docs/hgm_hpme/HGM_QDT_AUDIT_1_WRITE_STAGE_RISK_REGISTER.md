# HGM-QDT-AUDIT-1 — Write-Stage Risk Register

| ID | Severity | Area | Finding | Mitigation | Status |
|---|---|---|---|---|---|
| HGM-QDT-R1 | HIGH | payload materialization | HGM bridge payloads currently carry content_summary strings and metadata, while QDT/WM SystemWriteProposal requires a finite torch.Tensor content vector with a configured dimension. | Add a read-only materialization adapter stage that maps HGMBridgePayload / HGMEmbeddingRecord / SPCPProcedureEmbedding into bounded torch.Tensor proposals with dim negotiation and finite checks. | open |
| HGM-QDT-R2 | HIGH | slot identity | HGM SharedSlotLatticeHook.target_slot_id uses hgm_slot_* IDs, while WM SharedSlotRegistry.canonical_slot_id creates css-* IDs from namespace/local_slot/fingerprint. | Add a canonical-slot mapping adapter that treats HGM target_slot_id as local_slot_id and lets WM generate css-* canonical IDs, then records both IDs in lineage. | open |
| HGM-QDT-R3 | MEDIUM | depth mapping | HGM DepthLayer enum carries semantic depth labels; WM write APIs expect integer depth_index. | Freeze an explicit mapping D0_OBSERVATION=0 through D7_STRATEGIC=7 and validate all write plans before proposal generation. | open |
| HGM-QDT-R4 | MEDIUM | geometry mapping | HGM GeometryType enum must be converted to WM geometry_map strings. | Add a geometry-map translation table with fail-closed behavior for unsupported geometries. | open |
| HGM-QDT-R5 | HIGH | q-spin / QH code | HGM-4 generates qspin_placeholder_* values when q-spin is missing; WM QuantumHolographicStorage expects QH code/record semantics, not placeholder identity alone. | Introduce QSpin/QH-code conversion and require real QHCodeSchema compatibility before write enablement. | open |
| HGM-QDT-R6 | HIGH | write permission semantics | HGM write permission records are preview-only, while WM SystemCommitGate uses SystemWriteProposal.write_permission and require_write_permission at commit time. | Add explicit two-key handshake: HGM write execution approval + WM SystemWriteProposal.write_permission, both recorded in audit trace. | open |
| HGM-QDT-R7 | HIGH | rollback semantics | HGM rollback manifests currently describe rollback coverage, while WM SystemCommitGate.rollback_stack stores actual pre-commit snapshots after real commit. | Add a pre-commit snapshot reference protocol and require WM rollback_stack linkage in any live write stage. | open |
| HGM-QDT-R8 | MEDIUM | adapter detection | HGM adapter detection checks package paths/specs without heavy imports; this proves presence, not callable API compatibility. | Add a contract probe that inspects signatures for SystemWriteProposal, SystemCommitGate, SharedSlotStore, QuantumHolographicStorage, and QDTWorkingMemory without executing writes. | open |
| HGM-QDT-R9 | MEDIUM | dependency boundary | QDT/WM write surfaces depend on torch; HGM bridge/evaluation layers intentionally avoid hard torch dependency. | Add torch availability preflight and dependency-isolated adapter module; fail closed when torch is unavailable. | open |
| HGM-QDT-R10 | MEDIUM | trace redaction | HGM traces are redaction-compatible, but WM proposal metadata and shared-slot metadata can carry arbitrary dictionaries. | Redact metadata before SystemWriteProposal creation and store full sensitive payloads nowhere in HGM traces. | open |
| HGM-QDT-R11 | LOW | read-only status | HGM-4 through HGM-9 correctly preserve dry-run/evaluation-first behavior. | Keep this invariant in HGM-QDT-AUDIT-2 and any write-prep stage. | monitor |

## Write-stage gate recommendation

Do **not** enable live writes after this audit. The correct next step is a read-only write-preparation stage that creates:

1. tensor proposal materialization contract,
2. HGM slot hook to WM canonical slot mapping,
3. q-spin placeholder to QH code conversion contract,
4. explicit two-key permission handshake,
5. rollback snapshot binding plan,
6. signature-level QDT/WM contract probes.
