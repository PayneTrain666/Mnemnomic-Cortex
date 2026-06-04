# Exact REASON-1C Continuation Command

```text
DEV-FLOW RUN MNEMONIC-REASONING Stage REASON-1C — Integrate DepthIndexedSlotLattice into MANN SlotKV Reasoning Path

SOURCE OF TRUTH:
- REASON-1B pack:
  /mnt/data/mnemonic_reasoning_reason1b_wm_depth_integration_pack.zip
- Current WM/QDT source baseline:
  /mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip

PURPOSE:
Integrate slots × 8 depth capacity into the MANN reasoning scratchpad as a depth-indexed SlotKV bank while preserving existing MANN behavior by default.

DEV-FLOW STANDARDS:
- Deep implementation mandatory.
- Recalculate token budget first with actual figures.
- Create files, run tests, package ZIP, and print all generated contents.
- Split printout at file boundaries if needed.
- Apply patch phase and Full-Depth Adequacy Gate.

SAFETY:
- No destructive MANN replacement.
- No direct shared physical tensor storage with LTM.
- MANN depth lattice disabled/inert unless enabled by config.
- MANN writes remain shadow/proposal-only unless explicit gate permits mutation.

REQUIRED:
1. Read REASON-1B pack.
2. Create mann_depth_adapter.py.
3. Create mann_slotkv_depth_bank.py if needed.
4. Implement keys [S,8,K], values [S,8,V] MANN read/write proposals.
5. Implement hop-oriented read traces: selected slots, selected depths, depth entropy, support mass, confidence, disagreement.
6. Add tests for disabled default, enabled MANN depth read, SlotKV shape, no mutation default, trace serialization, and shared canonical IDs without shared tensors.
7. Create docs, tracker updates, ship-check, package ZIP, full file printout.
8. Provide exact REASON-1D command for LTM depth adapter integration.

```
