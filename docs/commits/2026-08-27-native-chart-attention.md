# Commit record — native chart attention

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Branch | `development-prototype` |
| Type | `feat` (prototype drop) |
| Scope | `wm`, `geometry`, `attention` |
| Status | Prototype / reviewable, not a production release |
| Parent | `6bd74a3` feat(prototype): mount geometry maps as native charts |

## Labels

Industry-style labels for this change (GitHub + review filters):

| Label | Kind | Why it applies |
|---|---|---|
| `type:feature` | type | IMA and pre-fusion attention now score/mix on native charts |
| `area:working-memory` | area | QDT-WM IMA, dual-fusion LTM/MANN/SPCP cross-attention |
| `area:geometry` | area | Log/exp at chart origin, geodesic query-key affinity |
| `status:prototype` | status | Development branch; not production-ready |
| `safety:qspin-inert` | safety | No QSPIN live routing/writes/commits |
| `risk:medium` | risk | New score/mix path; residual mix 0 is identity rollback |

## Summary

After maps were mounted as native charts, inter-manifold attention and
pre-fusion LTM/MANN/SPCP cross-attention still scored with Euclidean MHA /
cosine and mixed by adding charted vectors in ambient space. This drop makes
those attentions **read the charts**: geodesic / log-at-origin scores, then
weighted combination in the shared tangent and exp-map back. Residual mix `0`
stays identity.

## Motivation

Projecting tokens onto Poincaré or the sphere, then adding those points as if
they lived in R^D, is the wrong mix. Same-chart pairs should use the chart
metric. Different charts should compare log-maps at a shared origin
(tangent ≅ R^D), not ambient inner products of unrelated embeddings.

## Scope included

- Shared helpers in `geometry/chart_native.py`: `chart_origin`,
  `tangent_at_origin` / `retract_from_origin`, pairwise manifold affinity,
  query-key geodesic / log-at-origin scores, tangent transport.
- Inter-manifold attention (`wm_inter_manifold_attention.py`): native scores
  by default (`enable_native_chart_attention=True`,
  `native_chart_attention_mix=1.0`). Mix `< 1` blends MHA weights, then still
  transports in the tangent. Sequence residual is `mix_proj(mean(tangents))`.
  QDT flags: `enable_ima_native_chart_attention`,
  `ima_native_chart_attention_mix`.
- Pre-fusion LTM/MANN/SPCP cross-attention: same score/mix on retrieved keys
  via `score_and_mix_memory_on_charts`. Dual fusion combines the resulting
  tangent messages. QDT flags: `enable_prefusion_native_chart_attention`,
  `prefusion_native_chart_attention_mix`. Metrics `pfa_*`.
- Euclidean log/exp remains identity. Vector Grassmann/CP charts still fall
  back to the sphere (CP/Grassmann log/exp unused).
- Tests: IMA native path, pre-fusion identity/transport, Euclidean tangent
  mix equals weighted sum, QDT `pfa_*` metrics.

## Scope excluded

- Generated run artifacts (`prod7_outputs/`, `logs/`).
- One-shot source-rewrite helpers `tools/_fix_wm_nan_inplace.py` and
  `tools/_sanitize_wm_nan_guards.py`.
- Dual-quaternion SE(3) native charts; CP/Grassmann matrix log/exp.
- LTM triple-hybrid `_prefusion_specialization_exchange` (bank-readout MHA
  inside EnhancedTripleHybridMemory, not QDT dual-fusion pre-fusion).
- QSPIN live activation, shared-slot/LTM/MANN/QH writes, or commit execution.

## Tests

- `tests/test_wm_inter_manifold_attention.py`
- `tests/test_wm4a_cross_attention_dual_fusion.py`
- `tests/test_chart_native_geometry.py`
- `tests/test_wm_qd3a_module_contracts.py`
- `tests/test_wm2c_qdt_working_memory_assembly.py`
- `tests/test_wm4a_qdt_integration.py`
- `tests/test_wm_qd3a_qdt_attention_runtime_regression.py`
- `tests/test_wm_qd4a_external_memory_guards.py`

## Rollback

1. `git revert <this-commit>` on `development-prototype`.
2. Fast operational rollback without revert:
   - IMA: `enable_ima_native_chart_attention=False` or
     `inter_manifold_residual_mix=0` (token identity).
   - Pre-fusion: `enable_prefusion_native_chart_attention=False` or
     LTM/MANN/SPCP `residual_mix=0` (token identity).
   Residual mix `0` skips mixing into the token stream.

## Safety

QSPIN stays inert. Native chart attention does not write shared slots, LTM,
MANN, SPCP, or QH storage. Dual-fusion still uses the synthetic external-memory
query contract (read-only adapters).
