# HGM-0B Ship Check

## Status
PASS.

## Commands

```bash
python -m pytest tests/test_hgm_0b_probability_expander.py -q
python -m pytest tests/test_hgm_0a_foundation_types.py tests/test_hgm_0b_probability_expander.py -q
python -m compileall -q mnemonic_cortex/hypergraph_manifold
```

## Targeted result
13 passed.

## Compatibility result
21 passed across HGM-0A + HGM-0B tests.

## Compile result
compileall passed.

## Additive doctrine
No existing working_memory/QDT modules were modified. HGM-0B changes are isolated to `mnemonic_cortex/hypergraph_manifold/`, tests, docs, and release metadata. Cache directories are excluded from release comparison and packaging.

## Changed files

```json
{
  "added": [
    "docs/hgm_hpme/01_hgm_0b_probability_expander.md",
    "mnemonic_cortex/hypergraph_manifold/normalization.py",
    "mnemonic_cortex/hypergraph_manifold/probability_expander.py",
    "mnemonic_cortex/hypergraph_manifold/runtime_result.py",
    "mnemonic_cortex/hypergraph_manifold/scenario_extraction.py",
    "release/hgm_0b/changed_files.json",
    "release/hgm_0b/compatibility_pytest_output.txt",
    "release/hgm_0b/compile_output.txt",
    "release/hgm_0b/manifest.json",
    "release/hgm_0b/pytest_output.txt",
    "release/hgm_0b/ship_check.md",
    "tests/test_hgm_0b_probability_expander.py"
  ],
  "modified": [
    "mnemonic_cortex/hypergraph_manifold/__init__.py",
    "mnemonic_cortex/hypergraph_manifold/enums.py"
  ],
  "deleted": [],
  "ignored": [
    ".pytest_cache",
    "__pycache__",
    "*.pyc"
  ]
}
```

## Known limits
- Top-k extraction returns probability-cell candidates; scenario hyperedge binding is deferred to HGM-1.
- Runtime is dependency-light; torch/numpy adapters are future optional work.
- Token-based expansion treats `MutationToken.probability` as bounded probability, not logits.
