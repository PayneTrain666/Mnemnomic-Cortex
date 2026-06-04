# HGM-QDT-WRITE-PREP-4 — Real Contract Object Construction Dry-Run

## Purpose

WRITE-PREP-4 adds a dry-run layer for constructing real QDT/WM contract-object previews, defining a synthetic CommitGate adapter boundary, and planning rollback snapshot binding.

The stage remains non-mutating.

## What it adds

- `qdt_real_contract_object_dryrun.py`
- `qdt_synthetic_commitgate_adapter_boundary.py`
- `qdt_rollback_snapshot_binding_plan.py`
- `hgm_qdt_write_prep4_pipeline.py`
- `hgm_qdt_write_prep4_result.py`

## Safety boundary

WRITE-PREP-4 may construct `SystemWriteProposal` objects locally for validation previews, but it does not stage or commit them.

It never calls:

- `SystemCommitGate.stage`
- `SystemCommitGate.commit`
- `SharedSlotStore.write_slot`
- `QuantumHolographicStorage.create_record`
- rollback stack mutation APIs

## Rollback binding

Rollback binding remains a plan. Synthetic rollback replay evidence can be linked to preview requirements, but real `SystemCommitGate.rollback_stack` snapshots are not captured or bound in this stage.

## Known limits

- Not live-write ready.
- Does not bind actual rollback snapshots.
- Does not write to QDT/WM storage.
- Does not mutate QDT/WM internals.
