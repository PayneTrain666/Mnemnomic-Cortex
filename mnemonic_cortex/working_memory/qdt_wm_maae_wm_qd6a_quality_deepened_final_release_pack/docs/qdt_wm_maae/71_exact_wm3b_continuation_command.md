# Exact WM-3B Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-3B — Evidence, Counterfactual, Conflict, Novelty, Stability, and Trace Attention

DEV-FLOW STANDARDS:
- Deep implementation is mandatory by default.
- Recalculate token budget first with token figures.
- Token budget recalculation is binding.
- If deep implementation cannot fit, split at clean source-file boundaries.
- Do not compress source files, tests, docs, or trackers.
- Apply DEV-FLOW PATCH PHASE before ship-check.
- Apply Full-Depth Adequacy Gate before finishing.

TOKEN BUDGET ENFORCEMENT:
- Declare target scope, minimum complete version, deep implementation version, estimated file/module count, expected doc count, expected test count, clean split points, selected split scope, and explicit out-of-scope items.
- If needed, split:
  - WM-3B.1 — evidence + trace attention
  - WM-3B.2 — counterfactual + conflict/quarantine attention
  - WM-3B.3 — novelty + stability attention + tests/docs/ship-check

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, true quaternion rotations, context geometry maps, curved core, QDTWorkingMemory assembly, MAAE lanes, shared-slot doctrine, and QH code requirements.

PAAMA-X REQUIREMENTS:
- Include trace governance, write-permission hooks, confidence/disagreement hooks, conflict/quarantine hooks, audit metadata, and policy-lane integration.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

Goal:
Implement advanced attention mechanisms on top of WM-3A MAAE.

Required:
1. Create/patch wm_evidence_attention.py.
2. Create/patch wm_trace_attention.py.
3. Create/patch wm_counterfactual_attention.py.
4. Create/patch wm_conflict_attention.py.
5. Create/patch wm_novelty_attention.py.
6. Create/patch wm_stability_attention.py.
7. Integrate outputs with WMMemoryAugmentedAttention or QDTWorkingMemory trace path.
8. Add tests for each attention mechanism and combined trace metadata.
9. Update tracker/deferred work.

Output:
- Token budget recalculation with token figures.
- Source files.
- Tests.
- Patches.
- Tracker updates.
- Deferred work updates.
- Patch phase summary.
- Ship-check.
- Full-Depth Adequacy Gate.
- Exact WM-4A continuation command.
```
