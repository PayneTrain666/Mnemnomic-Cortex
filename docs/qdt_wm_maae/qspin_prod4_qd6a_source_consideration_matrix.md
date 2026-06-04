# QSPIN-PROD-4-QD6A Source Consideration Matrix

Covered families: foundation/context, depth/transformer, attention, external memory, shared-slot/QH, commit/cortex/compatibility, guards, quality subsystem, QSPIN-0 through QSPIN-8, PROD-0, PROD-1, PROD-2, PROD-3, PROD-4, release/docs/tests/benchmarks, production caveats, runtime safety regression.

Direct PROD-4 use: QSPIN production configs, PROD-1 gates, PROD-2 simulation layers, PROD-3 active-dry-run contracts, and new synthetic payload/sandbox/regression contracts. Considered but not touched: live QD6A runtime modules including context_triplet_projector.py, wm_quantum_holographic_storage.py, wm_shared_slot_store.py, wm_external_memory_interfaces.py, wm_system_commit_gate.py, guard modules, attention lanes, quality subsystem, and compatibility/cortex integrations. Deferred: real runtime activation, real writes, real QH interference execution, real external memory adapters, production deployment.

Compatibility rule: QD6A wins over WM-7A. WM-7A remains historical-only. PROD-4 must preserve context/triplet, depth/transformer, attention, external memory, shared-slot/QH, guard, commit/cortex/compatibility, and quality semantics. Required evidence: tests, docs, release manifest, safety regression, no-mutation audit, synthetic sandbox isolation audit.
