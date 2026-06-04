# HGM-5 — Learned Hypergraph-Manifold Embedding Trainer, Bridge Evaluation Harness, and Integration Scoring Suite

## Purpose

HGM-5 is the first evaluation and learning-preparation layer for the HGM/HPME stack. It is evaluation-first and read-only: it does not mutate QDT/WM internals, does not write memory, does not execute robotics actions, and does not require Torch or NumPy.

## What HGM-5 Adds

- Deterministic baseline HGM embeddings for bridge records.
- Bridge payload quality metrics.
- Shared slot-lattice hook quality metrics.
- Execution-preview quality metrics.
- Integration readiness scoring.
- A high-level `build_hgm5_embedding_evaluation(...)` entry point.

## Learned Embedding Trainer Scaffold

The `embedding_trainer.py` module implements a deterministic baseline scaffold. It converts supported HGM-4 records into fixed-length vectors using a stable hash projection. This is deliberately not a production learned trainer. It creates a stable evaluation substrate that future learned trainers can replace without changing the surrounding result contracts.

Supported records:

- `HGMBridgePayload`
- `SharedSlotLatticeHook`
- `TraceSafeMemoryPlan`
- `BridgeExecutionPreview`

Unsupported records degrade safely with warning-only validation results.

## Why Evaluation-First

HGM-5 sits between dry-run bridge planning and future write-capable memory integration. At this point, the safe move is to measure bridge quality and integration readiness before any write-permission stage. The layer therefore focuses on:

- deterministic scoring,
- traceability,
- redaction-compatible metrics,
- bounded record handling,
- optional dependency fallback,
- and compatibility with HGM-0A through HGM-4.

## Bridge Evaluation Metrics

`evaluate_bridge_payload_quality(...)` scores payloads using:

- source ID completeness,
- depth-layer validity,
- geometry validity,
- q-spin presence,
- content-summary presence,
- trace ID presence,
- redaction compatibility.

`evaluate_slot_hook_quality(...)` scores hooks using:

- source ID completeness,
- target slot ID stability,
- depth-layer validity,
- geometry validity,
- q-spin presence,
- dry-run safety,
- confidence range.

`evaluate_execution_preview_quality(...)` scores previews using:

- preview-only behavior,
- write-intent blocking where needed,
- planned operation count visibility,
- validation status,
- no-execution guarantee.

## Integration Scoring Suite

`score_hgm_integration_readiness(...)` combines embeddings, payload metrics, hook metrics, and optional preview metrics into deterministic `IntegrationScore` records and an aggregate readiness value in `[0, 1]`.

## Optional Dependency Fallback

HGM-5 accepts options for optional NumPy/Torch detection, but it does not depend on either package. If optional dependencies are unavailable, HGM-5 continues using pure-Python deterministic scoring.

## Future Connection

HGM-5 prepares the ground for HGM-6:

> Write-Permission Gate, Transactional Memory Integration Preview, and Rollback-Safe QDT/WM Commit Plan.

HGM-5 does not grant write permission. It only creates embeddings and evaluation results that later stages can use for readiness gates.
