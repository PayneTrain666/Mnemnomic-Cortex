# HGM-6 — Write-Permission Gate, Transaction Preview, and Rollback-Safe Commit Plan

HGM-6 adds the first explicit write-permission and transactional commit-preview layer for HGM/HPME. It remains preview-only: no QDT/WM writes are executed, no working-memory internals are mutated, and no robotics or hardware actions are emitted.

## Purpose

HGM-6 converts HGM-4 `TraceSafeMemoryPlan` slot hooks into preview-only transaction operations, builds rollback coverage records, scores commit readiness, and returns a typed `HGM6WritePermissionResult`.

## Preview-only safety model

The default write state is denied:

- `requested=False`
- `granted=False`
- `dry_run=True`
- `preview_only=True`

Even if a caller sets `granted=True`, HGM-6 does not execute writes. The grant only affects readiness scoring and preview status for a later explicit write-execution stage.

## Transaction operation preview

`build_transaction_operation_previews(...)` converts `TraceSafeMemoryPlan.slot_hooks` into `TransactionOperationPreview` records. Each preview operation includes source payload ID, target slot ID, depth layer, geometry type, q-spin signature, allowed/blocked state, and a trace ID.

Operation ordering is deterministic by depth, target slot, source record, and hook ID.

## Rollback manifest

`build_rollback_manifest(...)` builds one `RollbackOperation` for each preview operation. The manifest is complete only when every allowed preview operation has rollback coverage.

Rollback records are references only; they do not capture or mutate live QDT/WM state.

## Commit-readiness scoring

`score_commit_readiness(...)` combines:

- QDT/WM adapter availability from the memory plan
- dry-run/write-intent safety
- planned operation coverage
- rollback completeness
- HGM-5 integration readiness score, or conservative fallback
- explicit write permission state

A missing HGM-5 score uses a conservative fallback. Adapter unavailability, missing rollback coverage, invalid memory plans, or denied write permission block readiness.

## High-level entry point

`build_hgm6_write_permission_gate(...)` returns:

- `WritePermissionState`
- `TransactionCommitPreview`
- merged validation results
- trace records

## Known limits

HGM-6 does not perform live commits. Actual write execution requires a later explicit permission stage, currently planned as HGM-7.
