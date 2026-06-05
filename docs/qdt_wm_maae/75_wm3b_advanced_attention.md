# WM-3B Evidence, Counterfactual, Conflict, Novelty, Stability, and Trace Attention

## Source-integrity audit before WM-3B

| module | exists before WM-3B |
|---|---|
| `wm_evidence_attention.py` | False |
| `wm_trace_attention.py` | False |
| `wm_counterfactual_attention.py` | False |
| `wm_conflict_attention.py` | False |
| `wm_novelty_attention.py` | False |
| `wm_stability_attention.py` | False |

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_evidence_attention.py
mnemonic_cortex/working_memory/wm_trace_attention.py
mnemonic_cortex/working_memory/wm_counterfactual_attention.py
mnemonic_cortex/working_memory/wm_conflict_attention.py
mnemonic_cortex/working_memory/wm_novelty_attention.py
mnemonic_cortex/working_memory/wm_stability_attention.py
mnemonic_cortex/working_memory/wm_memory_augmented_attention.py
mnemonic_cortex/working_memory/qdt_working_memory.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm3b_advanced_attention_modules.py
tests/test_wm3b_advanced_attention_integration.py
```

## Implemented behavior

- Evidence-structured attention.
- Trace-aware attention.
- Counterfactual attention probe.
- Conflict-aware attention with quarantine metadata.
- Novelty/lightbulb attention.
- Stability-aware attention and repair guard.
- Integrated advanced attention traces into WMMemoryAugmentedAttention.
- QDTWorkingMemory trace path now exposes advanced attention metadata through MAAE trace payload.

## Explicit deferrals
- Actual do-not-reuse memory mutation is deferred to governance/write stages.
- Full LTM/MANN/SPCP dual fusion is deferred to WM-4A.
- Shared slot store is deferred to WM-4B.
