# REASON-2A Reasoning Controller Design

## Source and token budget

```json
{
  "source_info": {
    "active_source_pack": "/mnt/data/mnemonic_reasoning_latest_development_integration_pack.zip",
    "active_source_sha256": "415611feeb252100f49141614e6e627567ee228b27c1134591b00ce78e2a319f",
    "repair_log": [
      {
        "file": "mnemonic_cortex/reasoning_depth/shared_depth_slot_registry.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/mann_slotkv_depth_bank.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1c_mann_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/mann_depth_adapter.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1c_mann_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/ltm_depth_banks.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/ltm_depth_adapter.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1d_ltm_depth_integration_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_capacity_validation.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_integration_readiness.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/reasoning_depth/depth_lattice_benchmarks.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "forced_authoritative_copy"
      },
      {
        "file": "mnemonic_cortex/working_memory/mann_depth_integration.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "missing_file_copy"
      },
      {
        "file": "mnemonic_cortex/working_memory/ltm_depth_integration.py",
        "source_pack": "/mnt/data/mnemonic_reasoning_reason1e_capacity_validation_pack.zip",
        "mode": "missing_file_copy"
      }
    ],
    "wm_depth_controller_exists": true,
    "mann_depth_adapter_exists": true,
    "ltm_depth_adapter_exists": true,
    "shared_depth_slot_registry_exists": true,
    "capacity_validation_exists": true,
    "readiness_exists": true
  },
  "token_budget": {
    "target_scope": "REASON-2A reasoning controller, orchestration trace, shadow consolidation gate, tests, docs, tracker, package, latest integration update, and print sequence.",
    "minimum_complete_version_tokens_est": 14000,
    "deep_implementation_version_tokens_est": 39000,
    "generated_content_total_tokens_est": 90000,
    "max_safe_response_budget_tokens_est": 18000,
    "fits_single_response": false,
    "binding_split_decision": "Implementation/tests/package completed in one execution; printout split into REASON-2A-PRINT-P1/P2/P3.",
    "estimated_file_module_count": {
      "new_source_files": 3,
      "patched_source_files": 1,
      "new_tests": 7,
      "new_docs": 6,
      "updated_tracker": 1,
      "release_manifest": 1
    },
    "selected_split_scope": "Full REASON-2A implementation + package; source printout begins in final response.",
    "clean_split_points": [
      "source-integrity audit",
      "historical source repair",
      "export repair",
      "orchestration trace",
      "consolidation gate",
      "reasoning controller",
      "tests",
      "docs/tracker/ship-check",
      "package bundles",
      "print sequence"
    ],
    "explicit_out_of_scope": [
      "advanced policy/routing strategy expansion (REASON-2B)",
      "permanent consolidation commit execution",
      "production external memory-store activation",
      "model weight mutation",
      "optimizer mutation",
      "destructive WM/MANN/LTM replacement",
      "unbounded reasoning loops",
      "fake production-complete claim",
      "fake quantum hardware backend"
    ]
  }
}
```

## Design

REASON-2A adds the first safe controller over the slots × 8 depth substrate.

Created modules:

- `reasoning_orchestration_trace.py` — bounded JSON-safe orchestration trace.
- `consolidation_gate.py` — shadow/proposal-only gate with conflict/quarantine hooks.
- `reasoning_controller.py` — disabled-by-default WM→MANN→LTM orchestration loop.

The controller is additive. It does not replace QDTWorkingMemory, MANN, or LTM. Enabled mode performs bounded reads and creates only shadow consolidation proposals.
