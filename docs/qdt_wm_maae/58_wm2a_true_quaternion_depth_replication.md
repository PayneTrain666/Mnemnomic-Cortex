# WM-2A True Quaternion Depth Replication

## Source files created/updated

```text
mnemonic_cortex/working_memory/wm_quaternion_depth.py
mnemonic_cortex/working_memory/__init__.py
tests/test_wm2a_quaternion_depth_replication.py
```

## Implemented behavior

- Replaced scalar-only depth modulation with true packed 3D quaternion rotations.
- Preserved [B,Z,T,3,D] output contract.
- Preserved triplet axis:
  - anchor
  - direction
  - phase
- Supports D % 3 remainder dimensions safely.
- Adds quaternion normalization, conjugate, Hamilton product, and vector rotation helpers.
- Adds trainable per-depth/per-triplet quaternion parameters.
- Adds identity-safe affine trim per depth/triplet.
- Adds depth consistency report.
- Adds explicit dual-quaternion/spatial hook placeholder without claiming completion.

## Deferred from WM-2A

- Dual-quaternion SE(3) transport is intentionally not implemented here.
- Depth-specific addressing is still WM-2B.
- Full memory-augmented attention remains WM-3A.
