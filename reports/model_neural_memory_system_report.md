# Comprehensive Neural Memory System Report

## Scope

This report evaluates the current neural architecture and memory-system design across:

- Working Memory (QDT-WM stack)
- Long-Term Memory (triple-hybrid runtime + LTM package stack)
- MANN / reasoning-depth integration
- Shared-slot and geometry/depth infrastructure

It also documents unusual hidden behavior, recently implemented structural refactors, and remaining alignment gaps.

## Executive Summary

The model currently contains a sophisticated multi-memory system with two major runtime fabrics:

1. `EnhancedMnemonicCortex` + `EnhancedTripleHybridMemory` (primary runtime path)
2. `QDTWorkingMemory` dual-fusion path (WM with LTM/MANN/SPCP adapters)

After this refactor, the primary LTM runtime now supports explicit 4-system doctrine semantics:

- `hg_episodic`
- `cgmn_semantic`
- `spatial_topological`
- `procedural_spcp`

with compatibility aliases preserved and a first-class procedural SPCP bank added.

The system is advanced and unique in combining:

- quaternion-depth tensor processing
- manifold-specific memory routing
- multi-bank fusion across memory families
- shared-slot doctrine and QH trace infrastructure

## Current WM Structure (QDT-WM)

Primary file: `mnemonic_cortex/working_memory/qdt_working_memory.py`

### Core WM pipeline

Input `[B,T,D]` is processed through:

1. Curved resonant WM core
2. Triplet projection
3. Quaternion depth replication `[B,Z,T,3,D]`
4. Intra-depth transformer (temporal)
5. Cross-depth transformer
6. Depth adapters
7. Depth-specific slot addressing
8. MAAE + transformer stack
9. Dual-fusion cross-memory attention (`ltm`, `mann`, `spcp`)
10. Optional cross-model attention stack with external context
11. Depth fusion + residual composition

### Why unique

- Depth is explicit and quaternionized, not only positional.
- Memory augmentation is lane-aware and multi-source.
- Write path is governed by commit-gate semantics (stage/evaluate/commit) with QH metadata hooks.

### Hidden / non-obvious behavior

- Read path writes a lightweight shared-slot anchor and QH record.
- Several quality contracts and guard layers are present as metadata-enforced doctrine.
- External memory banks default to synthetic adapters unless explicitly attached.

## Current LTM Structure

### A. Primary runtime: `EnhancedTripleHybridMemory`

Primary file: `mnemonic_cortex/triple_hybrid.py`

Banks now present:

- HG episodic (`hg`)
- CGMN semantic (`cgmn`)
- Curved legacy bank (`curved`)
- Procedural SPCP (`spcp`) **new first-class runtime bank**
- Spatial topological (`spatial`) when enabled

Additional stores:

- Consolidated LTM bank
- Neural field memory
- Per-bank quantum holographic slot banks

### B. LTM package stack: `TripleHybridLTM`

Primary file: `mnemonic_cortex/ltm/ltm_system.py`

This stack now supports:

- `hg`
- `cgmn`
- `spatial`
- `procedural` **newly added here**

on one shared value store with depth routing and geometry charts.

## Current MANN / Reasoning-Depth Structure

Primary path files:

- `mnemonic_cortex/reasoning_depth/mann_ltm_shared_slot_geometry.py`
- `mnemonic_cortex/reasoning_depth/ltm_depth_banks.py`
- `mnemonic_cortex/reasoning_depth/reasoning_controller.py`

Key characteristics:

- 8-depth lattice doctrine for both MANN and LTM depth banks.
- Canonical 4-bank naming exists in reasoning-depth:
  - `hg_episodic`, `cgmn_semantic`, `spatial_topological`, `procedural_spcp`
- Shared-geometry route can sync to shared-slot store when configured.
- Many paths are shadow/proposal-oriented by default unless explicitly enabled.

## Geometry Maps and Quaternion-Depth Provisioning

### Implemented provisioning

In `EnhancedTripleHybridMemory`:

- Added default depth geometry maps per canonical memory type:
  - episodic
  - semantic
  - spatial
  - procedural
- Added geometry mounting APIs:
  - `mount_geometry_map()`
  - `mount_geometry_maps()`
- Added structure APIs:
  - `structure_memory_entries()`
  - `write_structured_memory()`
- Added model structure API:
  - `describe_memory_structure()`

In `EnhancedMnemonicCortex`:

- Added forwarding APIs:
  - `mount_memory_geometry_maps()`
  - `describe_memory_system_structure()`
  - `structure_memory_entries()`
  - `write_structured_memory()`

### Depth assignment behavior

Structured-memory APIs now support quaternion-based depth assignment by deriving a depth index from quaternion-norm features and mapping that depth to a geometry chart entry.

## Implemented Structural Refactors in This Pass

### 1) Canonical 4-memory semantics in runtime LTM

- Added first-class `procedural_spcp` runtime bank to `EnhancedTripleHybridMemory`.
- Added canonical alias handling:
  - `hg_episodic`
  - `cgmn_semantic`
  - `spatial_topological`
  - `procedural_spcp`

### 2) Fusion and routing support for procedural bank

- SPCP now participates in:
  - bank instantiation
  - router feature extraction
  - fusion bank set
  - read/write and QH recording
  - topology mutation hooks
  - external context attention path

### 3) Adapter alignment

- `wm_triple_hybrid_ltm_adapter.py` updated to read canonical semantic/procedural/spatial bank outputs.

### 4) LTM package extension

- `mnemonic_cortex/ltm/config.py` extended with `DEFAULT_PROCEDURAL_DEPTH_CHART` and `procedural_depth_chart`.
- `mnemonic_cortex/ltm/ltm_system.py` extended with a procedural subsystem.

## Hidden/Unseen Aspects Worth Calling Out

- Multiple memory fabrics coexist (`memory` shared-slot subsystem and WM-local shared-slot registry), and adapter compatibility matters.
- Runtime has both legacy and doctrine naming modes; alias normalization is essential.
- There are dual episodic pathways:
  - HG bank in triple-hybrid
  - dedicated `HGEpisodicLTM` sidecar
- Some manifold names are doctrinal labels and may map internally to supported geometry operators.

## Design Quality Assessment

### Strengths

- Rich memory specialization across episodic/semantic/spatial/procedural lanes.
- Explicit depth geometry doctrine and strong traceability.
- Advanced fusion with context-aware, policy-driven pathways.
- Strong test coverage around wiring and integration.

### Complexity risks

- Parallel architecture surfaces can drift if not governed by one canonical profile/schema.
- Legacy curved bank and doctrinal procedural bank coexist; semantics can blur without explicit policy.
- Shared-slot unification across all runtime pathways still requires strict adapter contracts.

## Remaining Recommended Alignment Work

1. Fully align `reasoning_depth` live routing policy with canonical 4-bank names for all tasks.
2. Standardize one authoritative shared-slot interface layer used by WM/LTM/MANN adapters.
3. Add explicit per-task bank preference routing (episodic/semantic/spatial/procedural) in reasoning controller strategy.
4. Add dedicated procedural-SPCP manifold operator path beyond curved-bank approximation for stricter doctrinal fidelity.

## Validation Notes

Targeted integration tests were run and pass for updated wiring surfaces:

- memory wiring integration
- LTM package wiring
- unified config loader
- checkpoint optional subsystems

This confirms backward compatibility and successful integration of new bank semantics and structure APIs.
