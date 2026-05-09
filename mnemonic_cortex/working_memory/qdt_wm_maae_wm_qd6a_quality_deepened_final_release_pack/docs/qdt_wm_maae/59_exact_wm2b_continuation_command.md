# Exact WM-2B Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-2B — Quaternion Depth Transformer and Depth-Specific Addressing

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

PATCH / UPGRADE TRIGGER STANDARD:
- Log every defect, shallow area, drift, missing carryover item, and REDO requirement in the tracker.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, quaternion rotations, geometry maps, shared-slot doctrine, QH codes, and shadow-write doctrine.

PAAMA-X REQUIREMENTS:
- Include trace governance, confidence/disagreement hooks, conflict/quarantine hooks, and audit metadata.

SOURCE QUALITY STANDARD:
- No fake done modules. Include typed config, forward contract, shape checks, traces, stability hooks, tests, and integration notes.

Goal:
Implement Quaternion Depth Transformer and depth-specific addressing over [B,Z,T,3,D] replicated WM state.

Required:
1. Patch wm_intra_depth_transformer.py.
2. Patch wm_cross_depth_transformer.py.
3. Implement transformer processing for [B,Z,T,3,D].
4. Implement depth-specific addressing using per-depth curvature and geometry context.
5. Integrate QuaternionDepthReplicator with depth transformers.
6. Add trace outputs for intra-depth and cross-depth flow.
7. Add stability checks for shape, finite tensors, and depth consistency.
8. Add tests for intra-depth flow, cross-depth flow, depth-specific addressing, and no NaN/Inf.
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
- Exact WM-3A continuation command.
```
