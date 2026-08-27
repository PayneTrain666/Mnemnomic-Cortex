# Commit record — chart fusion policy testbed

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Branch | `development-prototype` |
| Type | `feat` (opt-in testbed) |
| Scope | `wm`, `fusion`, `geometry` |
| Status | Prototype testbed for finetuning / recipe search |
| Parent | `5608435` feat(prototype): score and mix attention on native charts |

## Labels

| Label | Kind | Why it applies |
|---|---|---|
| `type:feature` | type | Opt-in scenario/chart fusion policy and pre-fusion handoff |
| `area:working-memory` | area | Dual-fusion mix of WM/LTM/MANN/SPCP tangent messages |
| `status:prototype` | status | Testbed; default off |
| `safety:qspin-inert` | safety | No QSPIN live routing/writes/commits |
| `risk:low` | risk | Opt-in; gate 0 is the hardcoded prior |

## Summary

Dual fusion still used four frozen scalars (WM 0.55, LTM 0.18, MANN 0.18,
SPCP 0.09) even after pre-fusion returned native-chart tangent messages. This
drop adds an **opt-in** policy testbed that supersedes that mix with:

1. Hardcoded scenario role priors (one recipe per context map).
2. Chart-histogram priors from each map's `geometry_by_depth` × `depth_weights`.
3. Residual trainable logits (global + per-map + mean per-chart).
4. Runtime condition overlays (disagreement → WM, low confidence → LTM).

`weights = softmax(log(prior) + gate * learned + condition_mix * conditions)`

Gate `0` is the hardcoded recipe. Policy off is the historical four-weight mix.

## How to enable

```python
QDTWorkingMemoryConfig(
    ...,
    enable_chart_fusion_policy=True,
    chart_fusion_gate_init=0.0,       # evaluate hardcoded priors
    chart_fusion_condition_mix=0.15,  # runtime overlays
)
```

Finetune: `dual_fusion.fusion_policy.open_for_finetune(0.1)` then optimize
`finetune_parameter_groups()`. Log `snapshot_recipe()` for outer-loop search.
Do not train logits at gate `0` (no gradient through the residual).

## Linked pre-fusion handoff

Native chart attention already mixed retrieved keys in the tangent, but dual
fusion still read unlabeled `memory_context` and scraped chart names from
traces. That is not a fusion contract.

Opt-in `enable_prefusion_handoff` (implied by `enable_chart_fusion_policy`)
makes LTM/MANN/SPCP emit a `PreFusionHandoff`: `space=tangent_at_origin`,
`query_chart`, `key_charts`, and the tangent tensor. WM is wrapped as an
explicit Euclidean tangent (identity). Dual fusion mixes those payloads
instead of ambient vectors. Module: `wm_prefusion_handoff.py`. Metric: `pfh_*`.

## Scenario role priors (wm, ltm, mann, spcp)

| Map | Mix | Why |
|---|---|---|
| literal | 0.70 / 0.12 / 0.10 / 0.08 | Stay on current tokens |
| hierarchical | 0.28 / 0.42 / 0.18 / 0.12 | LTM trees / hyperbolic |
| temporal | 0.30 / 0.18 / 0.22 / 0.30 | Torus + SPCP routines |
| spatial_mechanical | 0.32 / 0.12 / 0.40 / 0.16 | MANN hops / SE(3) / quat |
| symbolic_mathematical | 0.28 / 0.38 / 0.16 / 0.18 | LTM + CP / phase |
| procedural | 0.22 / 0.14 / 0.26 / 0.38 | SPCP workflows |
| conflict_verification | 0.34 / 0.36 / 0.18 / 0.12 | WM evidence + LTM facts |
| creative_synthesis | 0.24 / 0.22 / 0.28 / 0.26 | Even analogical blend |
| policy_governance | 0.40 / 0.32 / 0.12 / 0.16 | WM + LTM, low hop drift |
| quantum_holographic | 0.26 / 0.30 / 0.20 / 0.24 | Binding across systems |

Unknown maps fall back to the historical 0.55 / 0.18 / 0.18 / 0.09 mix.
Final prior is `0.60 * role + 0.40 * chart_histogram`.

## Tests

- `tests/test_wm_chart_fusion_policy.py`
- `tests/test_wm_prefusion_handoff.py`
- `tests/test_wm4a_cross_attention_dual_fusion.py`
- `tests/test_wm4a_qdt_integration.py`

## Scope excluded

- Automatic outer-loop recipe search (snapshot is the hook; no live explorer).
- QSPIN live activation or extra store writes.
- Changing the default dual-fusion path (still the four scalars unless opt-in).

## Rollback

Leave `enable_chart_fusion_policy=False` and `enable_prefusion_handoff=False`
(default). Or set `chart_fusion_gate_init=0` and `chart_fusion_condition_mix=0`
to freeze at the hardcoded prior without learned residuals.

## Safety

QSPIN stays inert. The policy only reweights already-read tangent messages.
