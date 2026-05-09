# Exact WM-1D Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-1D — Geometry-Aware Addressing and Bounded Associative Spread

DEV-FLOW STANDARDS:
- Deep implementation is mandatory by default.
- Do not reduce to minimal or mid-level unless explicitly requested.
- Recalculate token budget first with token figures.
- Token budget recalculation is binding, not decorative.
- If deep implementation cannot fit, split at clean source-file boundaries.
- Do not compress source files, tests, docs, or trackers to avoid splitting.
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
Implement geometry-aware addressing and bounded associative spread using CurvedSlotStateBank and CurvatureMetricPolicy.

Required:
1. Create geometry_aware_addressing.py.
2. Implement addressing score using content similarity, curved distance, phase compatibility, importance, confidence, trace reliability, and context geometry bias.
3. Create bounded_associative_spread.py.
4. Implement row-stochastic normalization, sparsity mask, spectral norm clamp, decay, entropy floor, and max spread steps.
5. Implement bounded curved Hebbian association update.
6. Integrate with CurvedResonantWMCore where appropriate.
7. Add tests for normalization, boundedness, sparse spread, and no NaN/Inf.
8. Update tracker/deferred work.

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
- Exact WM-1E continuation command.
```
