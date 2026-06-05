# Exact REASON-1B Continuation Command

```text
DEV-FLOW RUN MNEMONIC-REASONING Stage REASON-1B — Integrate DepthIndexedSlotLattice into Working Memory Controller

SOURCE OF TRUTH:
- REASON-1A pack:
  /mnt/data/mnemonic_reasoning_reason1a_depth_lattice_pack.zip
- Current context-compression extension:
  /mnt/data/qdt_wm_maae_context_compression_memory_pack.zip
- Current WM/QDT source baseline:
  /mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip

PURPOSE:
Integrate DepthIndexedSlotLattice into the active working-memory/controller path as an optional additive WM depth adapter while preserving QDTWorkingMemory behavior by default.

DEV-FLOW STANDARDS:
- Deep implementation mandatory.
- Recalculate token budget first with figures.
- Create files, run tests, package ZIP, and print all generated contents.
- Split printout at file boundaries if needed.
- Apply patch phase and Full-Depth Adequacy Gate.

SAFETY:
- No destructive QDTWorkingMemory replacement.
- No automatic permanent memory-store mutation.
- WM depth lattice disabled/inert unless enabled by config.
- Writes remain shadow-only unless explicit allow_mutation=True and write_permission=True.

REQUIRED:
1. Read REASON-1A pack.
2. Create wm_depth_adapter.py.
3. Create wm_depth_controller.py if needed.
4. Patch/add optional hooks into QDTWorkingMemory or compatibility wrapper only where source exists.
5. Integrate context compression candidates into Z3/Z5/Z7 proposal routes.
6. Add tests for disabled default, enabled depth read, shadow write proposals, trace emission, and QDT compatibility.
7. Create docs, tracker updates, ship-check, package ZIP, full file printout.
8. Provide exact REASON-1C command for MANN depth adapter integration.
```
