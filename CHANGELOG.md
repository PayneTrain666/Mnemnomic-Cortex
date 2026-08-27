# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version tags use [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where a tag exists. Prototype branch work is recorded under **Unreleased**.

## [Unreleased]

### Added

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
- Commit record: `docs/commits/2026-08-27-development-prototype.md`

[Unreleased]: https://github.com/PayneTrain666/Mnemnomic-Cortex/compare/f2e610b...development-prototype
