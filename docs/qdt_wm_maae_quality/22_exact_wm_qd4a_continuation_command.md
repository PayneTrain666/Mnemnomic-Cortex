# Exact WM-QD-4A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-4A — External Memory, LTM/MANN/SPCP Cross-Attention, Dual Fusion, Shared Slot Store, and Quantum-Holographic Storage Quality Deepening

Goal:
Use the WM-QD quality tooling to harden the external-memory, shared-slot, and QH-compatible storage layer.

Required:
1. Read the latest WM-QD-3A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_external_memory_interfaces.py
   - wm_ltm_cross_attention.py
   - wm_mann_cross_attention.py
   - wm_spcp_cross_attention.py
   - wm_dual_fusion.py
   - wm_shared_slot_registry.py
   - wm_shared_slot_store.py
   - wm_quantum_holographic_storage.py
3. Strengthen external memory response schemas, MANN trace visibility, fusion shape checks, shared-slot ownership/conflict metadata, QH code schema validation, interference checks, trace serialization, PAAMA-X write-permission metadata, and fallback behavior.
4. Preserve QDTWorkingMemory compatibility and prior attention/depth contracts.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-5A continuation command.

```
