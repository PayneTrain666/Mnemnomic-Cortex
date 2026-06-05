# HGM-2 — Manifold Chart Router, Geometry-Aware Distance Wrappers, and Depth-Layer Retrieval Bridge

## Purpose

HGM-2 adds the first geometry-aware routing layer for HGM/HPME. It consumes HGM-1 `BoundScenarioHyperedge` records, assigns them to compatible manifold charts, computes dependency-light geometry distance/similarity scores when coordinates are available, and emits depth-layer retrieval targets for later memory integration.

This release is additive and does not modify QDT, working-memory, SPCP, or existing Mnemonic Cortex runtime internals.

## Main modules

```text
mnemonic_cortex/hypergraph_manifold/hgm2_result.py
mnemonic_cortex/hypergraph_manifold/geometry_distance.py
mnemonic_cortex/hypergraph_manifold/manifold_router.py
mnemonic_cortex/hypergraph_manifold/depth_retrieval.py
```

## Geometry-aware distance wrappers

`compute_geometry_distance(point_a, point_b, geometry_type, config=None)` supports:

| Geometry | Method | Notes |
|---|---|---|
| `EUCLIDEAN` | L2 distance | `similarity = 1 / (1 + distance)` |
| `SPHERICAL` | cosine/angular distance | cosine is clamped to `[-1, 1]`; similarity is scaled to `[0, 1]` |
| `HYPERBOLIC` | safe Poincare-ball approximation | points must be inside the unit ball; invalid points fail closed |
| `TORUS` | wrapped circular L2 | values are interpreted modulo 1 |
| `COMPLEX_PROJECTIVE` | phase-invariant similarity approximation | real vectors can be alternating real/imag components; global phase equivalents stay close |
| `PRODUCT` | weighted component average | component metadata is preferred; missing component metadata degrades to Euclidean fallback with warning |

All wrappers return `GeometryDistanceResult` with `ValidationResult` and trace identity.

## Manifold chart routing

`route_hyperedges_to_manifold_charts(...)` assigns each HGM-1 bound scenario hyperedge to a manifold chart.

Inputs may be passed directly:

```python
route_hyperedges_to_manifold_charts(hyperedges, charts, config=config, routing_options=options)
```

Or through a `ManifoldRoutingInput` envelope:

```python
ManifoldRoutingInput(
    bound_hyperedges=(...),
    available_manifold_charts=(...),
    coordinates={...},
    geometry_preferences={...},
    depth_hints={...},
)
```

Routing uses:

1. chart validity,
2. optional geometry preference,
3. optional edge/chart coordinates,
4. deterministic score ordering,
5. stable chart-ID tie-breaks.

When coordinates are missing and `allow_missing_coordinates=True`, the router returns a structured fallback assignment with warning traces instead of crashing.

## Depth-layer retrieval bridge

`assign_depth_retrieval_targets(assignments, config=None, depth_options=None)` converts route assignments into retrieval targets.

Canonical HGM depth defaults are preserved:

| Depth | Meaning |
|---|---|
| D0 | observation |
| D1 | mutation |
| D2 | entity |
| D3 | relation |
| D4 | causal |
| D5 | procedural |
| D6 | counterfactual |
| D7 | strategic |

Default HGM-2 routing places generic scenario hyperedges into `D3_RELATION`. Procedural hyperedges route to D5, causal/conflict hyperedges to D4, and opportunity hyperedges to D7 unless explicit depth hints override this.

## Fallback behavior

- Empty hyperedge lists return an empty routing result with warning traces.
- Empty chart lists fail closed.
- Invalid charts fail closed.
- Unsupported geometry fails closed by default.
- Missing coordinates degrade safely if allowed.
- Missing metadata produces warnings or deterministic defaults, not crashes.

## How HGM-2 connects to HGM-3

HGM-2 produces chart assignments and depth retrieval targets. HGM-3 can consume those targets to connect procedural hyperedges into an SPCP procedural memory adapter, where action-sequence storage and robotics planning can use spherical/projective procedural structure.

HGM-2 is deliberately deterministic. Learned manifold embeddings, WM/QDT bridge adapters, and robotics action stores are deferred to HGM-3 and later stages.
