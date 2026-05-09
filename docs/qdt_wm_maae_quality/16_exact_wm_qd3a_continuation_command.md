# Exact WM-QD-3A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-3A — Memory-Augmented Attention, Geometry Scoring, Evidence, Trace, Counterfactual, Conflict, Novelty, and Stability Attention Quality Deepening

Goal:
Use the WM-QD quality tooling to harden the memory-augmented and advanced attention layer.

Required:
1. Read the latest WM-QD-2A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_retrieval_lanes.py
   - wm_geometry_scoring.py
   - wm_memory_augmented_attention.py
   - wm_geometry_linker.py
   - wm_evidence_attention.py
   - wm_trace_attention.py
   - wm_counterfactual_attention.py
   - wm_conflict_attention.py
   - wm_novelty_attention.py
   - wm_stability_attention.py
3. Strengthen candidate schemas, lane output validation, geometry score finite checks, attention boundedness, trace serialization, PAAMA-X metadata, conflict/quarantine hooks, and fallback behavior.
4. Preserve QDTWorkingMemory compatibility and prior depth contracts.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-4A continuation command.

```
