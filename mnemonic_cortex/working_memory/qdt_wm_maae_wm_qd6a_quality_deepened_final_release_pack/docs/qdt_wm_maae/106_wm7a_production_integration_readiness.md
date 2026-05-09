# WM-7A Production Integration Readiness

## Readiness status

**integration-ready with caveats**

## Ready now

- QDTWorkingMemory package-level use
- Compatibility-wrapper replacement in test/minimal cortex shells
- Read/process/write route smoke usage
- Trace inspection
- Shared-slot/QH/commit-gate metadata testing

## Not ready without more work

- Production persistent memory backend
- Real external LTM/MANN/SPCP adapters
- Real EnhancedMnemonicCortex source patch if source is not provided
- Large-scale training/performance profiling
- Distributed/concurrent commit-gate transaction guarantees

## Real-source integration required steps

1. Supply or locate the real EnhancedMnemonicCortex source file.
2. Import CortexWorkingMemoryIntegrationConfig and replace_cortex_working_memory.
3. Call replace_cortex_working_memory(self, CortexWorkingMemoryIntegrationConfig(...)) after dimensions are known.
4. Keep legacy_working_memory until parity confidence is proven.
5. Run all tests plus project-level integration tests.

## Carryover verification

```json
{
  "8_depth_slices": true,
  "triplets": true,
  "true_quaternion_rotations": true,
  "context_geometry_maps": true,
  "curved_core": true,
  "qdt_working_memory_assembly": true,
  "maae": true,
  "advanced_attention": true,
  "dual_fusion": true,
  "shared_slot_store": true,
  "qh_storage": true,
  "system_commit_gate": true,
  "cortex_integration_wrapper": true
}
```

## PAAMA-X verification

```json
{
  "trace_governance": true,
  "write_permission_hooks": true,
  "confidence_disagreement_hooks": true,
  "conflict_quarantine_hooks": true,
  "audit_metadata": true,
  "policy_lane_integration": true
}
```
