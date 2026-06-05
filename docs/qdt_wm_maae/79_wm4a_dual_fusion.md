# WM-4A LTM/MANN/SPCP Cross-Attention Interfaces and Dual Fusion Controller

## Source-integrity audit before WM-4A

| module | exists before WM-4A |
|---|---|
| `wm_external_memory_interfaces.py` | False |
| `wm_ltm_cross_attention.py` | False |
| `wm_mann_cross_attention.py` | False |
| `wm_spcp_cross_attention.py` | False |
| `wm_dual_fusion.py` | False |

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_external_memory_interfaces.py
mnemonic_cortex/working_memory/wm_ltm_cross_attention.py
mnemonic_cortex/working_memory/wm_mann_cross_attention.py
mnemonic_cortex/working_memory/wm_spcp_cross_attention.py
mnemonic_cortex/working_memory/wm_dual_fusion.py
mnemonic_cortex/working_memory/qdt_working_memory.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm4a_external_memory_interfaces.py
tests/test_wm4a_cross_attention_dual_fusion.py
tests/test_wm4a_qdt_integration.py
```

## Implemented behavior

- ExternalMemoryQuery / ExternalMemoryResponse contracts.
- SyntheticExternalMemoryBank contract adapter for local tests.
- LTM cross-attention.
- MANN cross-attention with required visibility:
  - pre-fusion output
  - per-hop attention
  - scratchpad tokens
  - confidence
  - disagreement
- SPCP procedural cross-attention.
- WMDualFusionController.
- QDTWorkingMemory trace path now includes dual_fusion.

## Explicit deferrals

- Real external LTM/MANN/SPCP adapters are deferred to integration stages.
- Shared canonical slot store is deferred to WM-4B.
- Quantum-holographic storage interface is deferred to WM-4C.
