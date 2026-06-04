# HGM-1 — Hyperedge Scenario Binder

HGM-1 converts HGM-0B `ScenarioCandidate` probability cells into bounded scenario hyperedges. The layer is deliberately deterministic, dependency-light, and additive.

## Runtime path

```text
HGM-0B ScenarioCandidate records
  -> bind_scenario_candidates
  -> BoundScenarioHyperedge records
  -> score_hyperedge_coherence
  -> detect_conflict_edges
  -> detect_opportunity_edges
  -> HGM1ScenarioGraphResult
```

## Grouping model

The binder groups candidates by configurable candidate fields:

```text
context_id, time_index, action_id, depth
```

This means candidates from the same context/time/action/depth become a first-pass scenario bundle. Later HGM releases can add learned grouping, geometry-aware similarity, or graph-neural binding.

## Coherence scoring

HGM-1 uses a deterministic heuristic score based on:

- average candidate probability
- hyperedge size
- shared context/action/time/depth metadata
- optional variable relation hints

Missing optional metadata creates warnings, not hard failure.

## Conflict graph

The conflict detector finds opposing mutation directions for the same variable inside a hyperedge. Direction can come from candidate metadata or from magnitude-bin names such as `up`, `down`, `positive`, or `negative`.

## Opportunity graph

The opportunity detector finds beneficial convergence bundles when at least two candidates in a coherent hyperedge carry `opportunity`, `beneficial`, or positive `utility` metadata.

## HGM-2 bridge

HGM-2 should route `BoundScenarioHyperedge` records into manifold charts and depth-layer retrieval surfaces. HGM-1 intentionally does not implement geometry kernels; it only creates clean hyperedge objects and audit traces for the next stage.
