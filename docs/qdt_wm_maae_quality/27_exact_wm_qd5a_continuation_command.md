# Exact WM-QD-5A Continuation Command

```text
DEV-FLOW RUN QDT-WM-MAAE Stage WM-QD-5A — System Commit Gate, Compatibility Wrapper, Cortex Integration, and QDT Write-Path Quality Deepening

Goal:
Use the WM-QD quality tooling to harden the system commit gate, compatibility wrapper, cortex integration layer, and QDT write/read integration path.

Required:
1. Read the latest WM-QD-4A quality pack.
2. Classify and remediate in-scope quality issues for:
   - wm_system_commit_gate.py
   - wm_compatibility_wrapper.py
   - wm_cortex_integration.py
   - qdt_working_memory.py write/read integration path
3. Strengthen commit proposal schemas, commit/reject/rollback/quarantine decisions, PAAMA-X write-permission enforcement, rollback trace safety, compatibility wrapper shape/finite checks, cortex migration template safety, and no-fake-real-source-patch guarantees.
4. Preserve QDTWorkingMemory compatibility and prior external-memory/attention/depth contracts.
5. Add/strengthen tests.
6. Update quality tracker/deferred register.
7. Run full tests.
8. Produce ship-check and exact WM-QD-6A continuation command.

```
