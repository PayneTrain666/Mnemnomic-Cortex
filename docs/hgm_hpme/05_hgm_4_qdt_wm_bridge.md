# HGM-4 — QDT/WM Bridge Adapter, Shared Slot-Lattice Hooks, and Trace-Safe Memory Integration

## Purpose

HGM-4 creates the first bridge-planning layer between the HGM/HPME stack and the existing Mnemonic Cortex QDT/working-memory ecosystem. It is deliberately **dry-run/read-only by default**. It produces typed payloads, shared slot-lattice hook contracts, adapter status records, and execution previews without mutating live memory internals.

## Why HGM-4 is dry-run/read-only by default

The bridge boundary is high risk because QDT/WM memory internals may contain live slot stores, depth maps, adapters, attention surfaces, and future learned state. HGM-4 therefore emits contracts and previews only. Any future write-capable bridge must be implemented in a separate explicit write-permission stage.

Default behavior:

- `dry_run=True`
- `write_intent=False`
- `allow_write_preview=False`
- no direct writes
- no hardware calls
- no network calls
- no live robotics-control execution

## Main records

### `BridgeAdapterStatus`

Reports whether expected QDT/WM module or filesystem paths are present. Detection uses `importlib.util.find_spec` and filesystem checks only; it does not import heavy runtime modules.

### `HGMBridgePayload`

A redacted, compact bridge payload converted from supported HGM records:

- `BoundScenarioHyperedge`
- `ManifoldRouteAssignment`
- `DepthRetrievalTarget`
- `ProceduralActionSequence`
- `SPCPProcedureEmbedding`
- `ProceduralMemoryRetrievalCandidate`
- `RoboticsPlanningActionOption`

### `SharedSlotLatticeHook`

A deterministic dry-run hook contract containing:

- source record ID
- target slot ID
- depth layer
- geometry type
- q-spin signature ID or placeholder
- dry-run flag
- write-intent flag
- confidence

### `TraceSafeMemoryPlan`

Combines adapter status, bridge payloads, shared slot hooks, validation, and trace records.

### `BridgeExecutionPreview`

Preview-only operation list. It never performs memory writes. If `write_intent=True` and `allow_write_preview=False`, it blocks the preview and records the reason.

## Shared slot-lattice hook contracts

Target slot IDs are deterministic hashes over the payload source type, source ID, geometry type, and q-spin signature. This makes hook generation idempotent and stable across repeated planning runs.

Example target slot form:

```text
hgm_slot_<depth>_<stable-hash>
```

## Depth-layer mapping

HGM-4 preserves depth metadata from source records when available. Procedural records default to:

```text
D5_PROCEDURAL
```

Unsupported or missing depth metadata degrades to deterministic defaults rather than crashing.

## Q-spin bridge placeholders

If source records do not carry a q-spin signature, HGM-4 generates a deterministic placeholder:

```text
qspin_placeholder_<stable-hash>
```

This preserves future compatibility with Q-spin alignment without pretending a real trained q-spin binding exists.

## Trace-safe memory integration

All bridge functions produce `TraceRecord` objects. Payload metadata is redacted for obvious secret-bearing keys such as:

- secret
- token
- api_key
- password
- credential
- private_key

This is not a full DLP system, but it prevents obvious accidental leakage in audit traces.

## Future write-permission stages

HGM-4 intentionally does not write into QDT/WM memory. A later write-capable stage should require:

1. explicit write permission,
2. adapter-specific schema verification,
3. rollback strategy,
4. isolated write transaction previews,
5. compatibility tests against QDT/WM runtime contracts.

## HGM-5 handoff

HGM-5 should add learned/evaluation layers around HGM/HPME:

- learned hypergraph-manifold embedding trainer,
- bridge evaluation harness,
- integration scoring suite,
- dry-run quality metrics for HGM bridge payloads and slot hooks.
