# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version tags use [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where a tag exists. Prototype branch work is recorded under **Unreleased**.

## [Unreleased]

### Added

- Native chart geometry from QDT/LTM/MANN maps (`geometry/chart_native.py`).
  Residual projection onto each depth's named manifold plus chart-metric
  scoring. Euclidean projection is identity; mix `0` is a no-op.
  QDT-WM projects `[B,Z,T,3,D]` after adapters (`ncg_*` metrics). LTM banks
  take blender priors and mixed chart distance from their 8-depth maps.
  MANN hops and shared-slot transforms use clean project+residual instead of
  tanh/sin warp. Grassmann vector charts fall back to the sphere.
- Inter-manifold attention for QDT-WM and cortex (`wm_inter_manifold_attention.py`).
  Monitors communications among geometry-map depths, LTM banks, MANN hops, SPCP,
  and optional PSLS views, then mixes them back with a residual gate.
  Mix `0` is identity. Does not write LTM/MANN/shared slots/QH or activate QSPIN.
- Parameter storage loop (PSLS) training path: post-optimizer consolidation,
  identity-safe residual gate, CPS-preserving resume.
- Trainable parameter CPS consolidation controls and capacity reporting helpers.
- Copy/reverse training metrics catalog, JSONL stream, and live dashboard.
- Copy-task nonfinite recovery (including complex tensors), HG complex FP32 path,
  and eval/CUDA hardening flags.
- Benchmark model capacity helpers and associated tests.

### Changed

- Depth-specific addressing blends cosine with `tanh(-chart_distance)` on
  non-Euclidean map depths when native chart mix is nonzero.
- Copy-task GPU trainer: QDT default fabric, capacity/metrics flags, PSLS wiring,
  and more durable checkpoint/eval behavior.
- WM attention modules sanitize nonfinite tokens instead of hard-failing the step.
- Cortex metrics now expose inter-manifold (`ima_*`) and PSLS gate diagnostics.

### Fixed

- Recurrent NaN / nonfinite logits on the copy-reverse path (AMP/`ComplexHalf`,
  missing complex snapshots, CUDA misaligned address during eval).
- `_LowRankDeltas.apply` no longer shadows `nn.Module.apply`.

### Security / safety

- QSPIN remains disabled/inert on this prototype path: no live routing, payload
  transfer, shared-slot writes, LTM/MANN/SPCP writes, QH writes, or commits.

### Notes

- Branch: `development-prototype`
- Commit records:
  - `docs/commits/2026-08-27-native-chart-geometry.md`
  - `docs/commits/2026-08-27-development-prototype.md`
- Local `pytorch_new/` checkout is ignored and no longer tracked as a gitlink.

[Unreleased]: https://github.com/PayneTrain666/Mnemnomic-Cortex/compare/f2e610b...development-prototype
