# HGM-3 — SPCP Procedural Memory Adapter

## Purpose

HGM-3 adds the first procedural-memory layer for HGM/HPME. It converts HGM-1/HGM-2 routed scenario records into typed procedural action sequences, builds dependency-light SPCP embeddings, stores/retrieves similar procedures, and emits advisory robotics planning options.

SPCP means **Spherical–Projective Conformal Procedural Memory**.

This stage is deliberately deterministic and dry-run safe. It does **not** execute robot actions, call hardware, mutate QDT/WM internals, or create live control policies.

## Data Flow

```text
HGM-1 BoundScenarioHyperedge
  + HGM-2 ManifoldRouteAssignment
  + HGM-2 DepthRetrievalTarget
        ↓
ProceduralActionSequence
        ↓
SPCPProcedureEmbedding
        ↓
ProceduralMemoryStoreResult
        ↓
ProceduralMemoryRetrievalResult
        ↓
RoboticsPlanningBridgeResult
```

## Action Sequences

A `ProceduralActionSequence` contains bounded `ActionPrimitive` records. Each primitive has:

- stable primitive ID
- action type
- finite parameter payload
- duration
- confidence
- optional frame/object/tool metadata

When explicit kinematic metadata is unavailable, HGM-3 generates deterministic generic action primitives from candidate/node IDs and emits warnings rather than crashing.

## Spherical Component

The spherical state is a normalized numeric action-state vector. It supports procedure comparison where the direction of the procedural state matters more than raw magnitude. Zero vectors fail closed unless fallback normalization is explicitly enabled.

## Complex-Projective Component

The projective state uses alternating real/imaginary components. Similarity is based on the absolute complex inner product, making it approximately invariant to global phase. This supports the projective equivalence concept: the same procedure can be represented with different global phase without losing identity.

## Conformal Warp

The conformal warp is a small deterministic bounded perturbation. Default maximum magnitude is `0.1`. It gives the embedding a controlled local deformation term without allowing runaway distortions.

## Procedural Memory

`store_procedural_sequences(...)` validates sequences, computes SPCP embeddings, and returns an immutable store result. It does not mutate a hidden global store.

`retrieve_similar_procedures(...)` uses SPCP similarity and deterministic tie handling to return bounded top-k candidates.

## Robotics Planning Bridge

`build_robotics_planning_options(...)` converts retrieved procedures into advisory planning options. These include:

- action primitive records
- expected goal
- risk score
- confidence
- explanation

The output is explicitly advisory only. It is **not** an actuator command and is not suitable for direct hardware execution.

## High-Level Entry Point

`build_hgm3_spcp_procedural_memory(...)` accepts a mapping or record bundle containing source hyperedges, route assignments, depth targets, and optionally a query sequence. It builds sequences, stores embeddings, retrieves similar procedures, and produces planning options.

## Safety / Reliability Notes

- fail-closed validation
- bounded action sequence length
- bounded action parameter count
- finite numeric checks
- deterministic retrieval tie handling
- trace generation and redaction support
- no network calls
- no external service dependency
- no hardware calls
- no mutation of QDT/WM internals

## Known Limits

- deterministic embeddings only; no learned procedural embedding yet
- advisory planning only; no live robotics control
- no QDT/WM runtime bridge yet
- no embodied policy execution

Those are deferred to HGM-4 and later stages.
