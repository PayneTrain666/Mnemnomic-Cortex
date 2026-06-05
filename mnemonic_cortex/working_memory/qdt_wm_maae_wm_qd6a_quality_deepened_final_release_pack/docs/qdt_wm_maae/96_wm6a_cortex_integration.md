# WM-6A EnhancedMnemonicCortex Integration, Migration Wrapper, and Working-Memory Replacement Patch

## Source-integrity audit before WM-6A

```json
{
  "wm_cortex_integration.py": false,
  "wm_compatibility_wrapper.py": false,
  "enhanced_mnemonic_cortex_source_hits": [],
  "real_cortex_source_available": false
}
```

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_compatibility_wrapper.py
mnemonic_cortex/working_memory/wm_cortex_integration.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm6a_compatibility_wrapper.py
tests/test_wm6a_cortex_integration.py
```

## Real cortex source status

Real `EnhancedMnemonicCortex` source found:
`False`

Source hits:
```text
No EnhancedMnemonicCortex source file found in this pack.
```

Patched cortex files:
```text
No real cortex source patch was applied because no complete EnhancedMnemonicCortex source file was present in the pack.
```

## Implemented behavior

- QDTWMCompatibilityWrapper.
- CortexWorkingMemoryIntegrationConfig.
- replace_cortex_working_memory().
- EnhancedMnemonicCortexQDTAdapter minimal integration shell for tests.
- migration_patch_template() for applying to real cortex source.
- Tests for read/process/write routing through QDTWorkingMemory.
- Tests for trace preservation.
- Tests for preserving old working memory reference.

## Explicit honesty note

This stage does not pretend to patch a real EnhancedMnemonicCortex source file if that source is absent from the package. It provides:
1. A tested migration function.
2. A tested compatibility wrapper.
3. A tested minimal cortex shell.
4. A concrete patch template for the real source file.
