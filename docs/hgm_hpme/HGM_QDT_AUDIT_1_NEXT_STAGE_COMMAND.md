# HGM-QDT-AUDIT-1 — Next Stage Command

```text
DEV-FLOW RUN HGM-QDT-WRITE-PREP-1 — Read-Only Proposal Materialization Contract, Slot-ID Mapping Plan, QH/Q-Spin Conversion Contract, and Rollback Snapshot Handshake

SOURCE OF TRUTH:
- Use the HGM-QDT-AUDIT-1 audited baseline as the current implementation baseline:
  /mnt/data/current-branch-HGM-QDT-AUDIT-1-audited.zip
- Preserve the HGM v0.1 release state:
  /mnt/data/current-branch-HGM-10-patched.zip
- Preserve the HGM/HPME package path:
  mnemonic_cortex/hypergraph_manifold/
- Preserve QDT/WM internals as read-only unless a later explicit write stage grants permission.

OBJECTIVE:
Create a read-only write-preparation contract layer that maps HGM bridge records toward QDT/WM write proposals without executing writes. This stage must define proposal materialization contracts, slot-ID mapping rules, q-spin/QH conversion contracts, rollback snapshot handshake requirements, and signature-level QDT/WM contract probes.

ACTIVE DEV-FLOW RUN COMBINED OPTIMAL GLOBAL STANDARDS:
1. Recalculate token budget first.
2. Execute in this order: CODE WALKTHROUGH → IMPLEMENT → AUDIT-PACK → PATCH NOW → SHIP-CHECK.
3. Preserve lineage from HGM v0.1 and HGM-QDT-AUDIT-1.
4. Dry-run/read-only only.
5. No live QDT/WM writes.
6. No mutation of working_memory/QDT internals.
7. No production write enablement.
8. All outputs must be typed, traceable, redacted, bounded, and fail-closed.
9. Tests and release artifacts are mandatory.

SCOPE — IMPLEMENTATION FILES:
Create additive files only, such as:

mnemonic_cortex/hypergraph_manifold/
  qdt_write_contract_probe.py
  qdt_proposal_materialization.py
  qdt_slot_mapping_plan.py
  qdt_qspin_qh_contract.py
  qdt_rollback_handshake.py
  hgm_qdt_write_prep_result.py

Add tests:
  tests/test_hgm_qdt_write_prep_1.py

Add docs:
  docs/hgm_hpme/HGM_QDT_WRITE_PREP_1.md

Add release metadata:
  release/hgm_qdt_write_prep_1/manifest.json
  release/hgm_qdt_write_prep_1/pytest_output.txt
  release/hgm_qdt_write_prep_1/compatibility_pytest_output.txt
  release/hgm_qdt_write_prep_1/compile_output.txt
  release/hgm_qdt_write_prep_1/changed_files.json
  release/hgm_qdt_write_prep_1/ship_check.md

REQUIRED OUTPUT:
- Proposal materialization contract for converting HGM payloads into bounded tensor proposal previews.
- Slot-ID mapping plan from hgm_slot_* to WM local_slot_id and css-* canonical IDs.
- Q-spin/QH conversion contract.
- Rollback snapshot handshake plan.
- Signature-level QDT/WM contract probe.
- No live writes.
- Next exact command.

NEXT EXACT COMMAND TO PRINT AFTER THIS STAGE:
DEV-FLOW RUN HGM-QDT-WRITE-PREP-2 — Dry-Run SystemWriteProposal Builder, CommitGate Preflight Harness, and Non-Mutating End-to-End Write Simulation

```
