# Exact WM-1C Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-1C — CurvedSlotState and CurvatureMetricPolicy

DEV-FLOW STANDARDS:
- Deep implementation is mandatory by default.
- Do not reduce to minimal or mid-level unless explicitly requested.
- Recalculate token budget first.
- Token budget recalculation is binding, not decorative.
- Include target scope, minimum complete version, deep implementation version, estimated file/module count, expected document count, expected test count, clean split points, selected split scope, and explicit out-of-scope items.
- If deep implementation cannot fit comfortably, split at clean source-file or document boundaries.
- Do not compress source files, tests, docs, audits, or trackers to avoid splitting.
- Each split must be independently deep, complete, and test-backed where tests are relevant.
- Include actual patch blocks, source files, docs, tests, acceptance criteria, deferred work, and ship-check.
- Apply DEV-FLOW PATCH PHASE before ship-check.
- Apply Full-Depth Adequacy Gate before finishing.
- If depth is insufficient, issue a REDO command for the affected split.
- Do not mark this stage complete until all required splits, tests, audits, patches, integration steps, tracker updates, and ship-checks are complete.

TOKEN BUDGET ENFORCEMENT:
- Recalculate token budget before implementation with token figures.
- Treat the token budget calculation as binding, not decorative.
- If deep implementation cannot fit, split at file boundaries.

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DEV-FLOW PATCH PHASE:
- Review outputs, tests, tracker items, and patch in-scope issues before ship-check.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, quaternion rotations, geometry maps, shared-slot doctrine, QH codes, and shadow-write doctrine.

PAAMA-X REQUIREMENTS:
- Include trace governance, write-permission hooks, confidence/disagreement hooks, conflict/quarantine hooks, and audit metadata where relevant.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

FULL-DEPTH ADEQUACY GATE:
- Confirm selected scope quality, tests, trackers, REDO status, stage completion, and next command.

Goal:
Implement CurvedSlotState and CurvatureMetricPolicy for the Curved Resonant WM Core.

Required:
1. Create curved_slot_state.py.
2. Define CurvedSlotState with slot_id, content, position, tangent, phase, curvature, importance, confidence, last_updated, trace_links.
3. Create curvature_metric_policy.py.
4. Implement global, per-slot, per-depth, and context-conditioned curvature.
5. Add curvature clamps and drift penalties.
6. Add shape/stability tests.
7. Update tracker and deferred work register.

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
- Exact WM-1D continuation command.
```
