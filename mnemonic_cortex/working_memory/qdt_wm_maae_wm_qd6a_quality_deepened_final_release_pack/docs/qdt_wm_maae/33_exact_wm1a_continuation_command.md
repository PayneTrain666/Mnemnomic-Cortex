# Exact WM-1A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-1A — Canonical EnhancedCurvedMemory Wrapper and Preservation Tests

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
- Recalculate token budget before implementation.
- Treat the token budget calculation as binding, not decorative.
- Declare whether the requested deep implementation fits in one response.
- If not, split immediately.
- Use clean split boundaries: documentation packs, source modules, test modules, integration patches, audit/benchmark packs.
- Do not summarize source files that should be printed or patched.
- Do not replace required source with prose.
- If a split cannot fit, create a sub-split.
- A parent split is not complete until all sub-splits are complete.

PATCH / UPGRADE TRIGGER STANDARD:
- If any prior section, file, design choice, test, interface, roadmap item, command, generated artifact, or architecture assumption is discovered to need a patch, upgrade, correction, refactor, expansion, or REDO, create or update the Patch and Upgrade Tracking Document immediately.
- Every issue must be recorded with tracker_id, affected stage/split, affected files/modules/docs/tests, reason, severity, status, required action, downstream impact, and completion evidence.
- Do not silently fix issues without logging them.
- Do not mark a stage complete while blocking tracker items remain incomplete.
- If a tracker item blocks correctness, issue a REDO or dedicated PATCH command.
- At the end of every stage/split, print the tracker status summary.

DEV-FLOW PATCH PHASE:
- Review all files, docs, tests, tracker items, prior split outputs, and current requirements.
- Identify defects, shallow areas, missing requirements, design drift, source/test mismatches, stale roadmap items, weak interface contracts, missing dimensional-depth carryover items, missing curved-WM preservation items, missing geometry-map/context-buffer items, missing MANN/LTM/SPCP fusion items, missing PAAMA-X hooks, missing shared-slot logic, and missing quantum-holographic storage logic.
- Log every issue into the Patch/Upgrade Tracker.
- Patch in-scope issues immediately.
- Defer out-of-scope issues explicitly.
- If a blocker is discovered outside the selected split scope, stop and issue the exact REDO/PATCH command.
- Print patch summary before ship-check.

DIMENSIONAL-DEPTH CARRYOVER REQUIREMENTS:
- Preserve 8 depth slices, triplets, quaternion depth rotations, geometry maps, trainable/warm-up weights, shared-slot doctrine, topology routing, geometry-specific scoring, QH depth codes, and shadow-write doctrine.

PAAMA-X REQUIREMENTS:
- Include policy lane, trace governance, write-permission hooks, confidence/disagreement hooks, conflict/quarantine hooks, and audit metadata in relevant modules.

SOURCE QUALITY STANDARD:
- No fake done modules.
- Skeletons are allowed only in architecture-lock stages.
- Once a module enters implementation phase, it must include typed config, forward contract, shape checks, trace hooks, stability hooks where relevant, unit tests, and integration notes.
- Fallbacks must be labelled and tracked.
- Direct replacement of original curved WM is prohibited unless wrapper parity is proven.

FULL-DEPTH ADEQUACY GATE:
- Confirm selected scope was implemented at deep-enough quality.
- List depth gaps.
- List tests present.
- List unresolved tracker blockers.
- Decide whether REDO is required.
- Declare whether split/stage is complete.
- Provide exact continuation command.

Goal:
Execute WM-1A: Canonical EnhancedCurvedMemory Wrapper and Preservation Tests.

Required:
1. Locate canonical EnhancedCurvedMemory source from current project code.
2. Replace fallback-only wrapper with canonical delegation path where available.
3. Preserve original encoder, curvature parameters, memory slots, importance, associative weights, addressing, activation spread, write path, and decoder.
4. Add compatibility tests against known old WM behavior.
5. Ensure no direct replacement of curved WM occurs.
6. Update tracker.

Output:
- Token budget recalculation.
- Source files.
- Tests.
- Tracker updates.
- Deferred work updates.
- DEV-FLOW PATCH PHASE summary.
- Acceptance criteria.
- Ship-check.
- Full-Depth Adequacy Gate.
- Exact WM-1B continuation command.
```
