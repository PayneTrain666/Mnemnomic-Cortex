# Commit record — native chart geometry

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Branch | `development-prototype` |
| Type | `feat` (prototype drop) |
| Scope | `wm`, `ltm`, `mann`, `geometry` |
| Status | Prototype / reviewable, not a production release |
| Parent | `6cc3aef` feat(prototype): add inter-manifold attention and stabilize copy-task training |

## Labels

Industry-style labels for this change (GitHub + review filters):

| Label | Kind | Why it applies |
|---|---|---|
| `type:feature` | type | Maps now create native manifold charts, not only labels/bias |
| `area:working-memory` | area | QDT-WM depth projection, addressing, dual-fusion queries |
| `area:ltm` | area | HG/CGMN/curved blender priors and chart-metric retrieval |
| `area:mann` | area | Hop projection and shared-slot chart transform |
| `area:geometry` | area | `chart_native` residual project + pairwise chart distance |
| `status:prototype` | status | Development branch; not production-ready |
| `safety:qspin-inert` | safety | No QSPIN live routing/writes/commits |
| `risk:medium` | risk | New residual mixers; default mix is modest; identity at 0 |

## Summary

Turn QDT, LTM, and MANN geometry maps into native charts: residual
`project_to_manifold` on the named chart, then score with that chart's metric
instead of cosine plus a tiny addressing bias.

## Motivation

Maps were schedules of chart names. Live QDT used them as labels and
`GEOMETRY_BIAS_TABLE` scalars. Shared-slot MANN warped with tanh/sin after a
project. This drop mounts the same maps as actual (vector-safe) manifold
geometry, with mix `0` as the identity rollback.

## Scope included

- Shared helper `geometry/chart_native.py`: residual project, pairwise chart
  distance/affinity, blender priors from chart histograms.
- QDT-WM: project `[B,Z,T,3,D]` after depth adapters; addressing mixes cosine
  with `tanh(-distance)` on non-Euclidean depths; dual fusion passes the map
  into LTM/MANN/SPCP. Defaults `enable_native_chart_geometry=True`,
  `native_chart_residual_mix=0.20`. Metrics `ncg_*`.
- LTM: bank maps drive GeometryBlender priors and mixed native distance on
  HG/CGMN/curved (mix `0.15`). `configure_native_chart_geometry` /
  `mount_geometry_map` keep banks in sync.
- MANN: hop scratchpad projection; shared-slot `_chart_transform` is residual
  project (not tanh/sin). Vector Grassmann/CP charts fall back to sphere.
- Tests: `tests/test_chart_native_geometry.py`.
- Housekeeping: ignore and untrack the broken `pytorch_new` gitlink.

## Scope excluded

- Generated run artifacts (`prod7_outputs/`, `logs/`).
- One-shot source-rewrite helpers `tools/_fix_wm_nan_inplace.py` and
  `tools/_sanitize_wm_nan_guards.py`.
- Dual-quaternion SE(3) native charts; CP/Grassmann log/exp (still unused).
- QSPIN live activation, shared-slot/LTM/MANN/QH writes, or commit execution.

## Tests

- `tests/test_chart_native_geometry.py`
- `tests/test_wm2b_depth_specific_addressing.py`
- `tests/test_wm2c_qdt_working_memory_assembly.py`
- `tests/test_wm4a_cross_attention_dual_fusion.py`
- `tests/test_wm_inter_manifold_attention.py`
- `tests/test_reason_shared_slot_mann_ltm_geometry.py`
- `tests/test_mann_ltm_manifold_shared_slot.py`
- `tests/test_manifold_utils.py`
- `tests/test_wm4a_qdt_integration.py`
- `tests/test_memory_wiring_integration.py`
- `tests/test_triple_hybrid_ltm_adapter_topk.py`
- `tests/test_wm_qd3a_qdt_attention_runtime_regression.py`

## Rollback

1. `git revert <this-commit>` on `development-prototype`.
2. Fast operational rollback without revert: set
   `native_chart_residual_mix=0` (QDT also accepts
   `enable_native_chart_geometry=False`); LTM
   `configure_native_chart_geometry(0.0)`; MANN
   `SharedGeometrySlotConfig(..., native_chart_residual_mix=0.0)`.
   Mix 0 is identity.

## Safety

QSPIN stays inert. Native chart projection does not write shared slots, LTM,
MANN, SPCP, or QH storage.
