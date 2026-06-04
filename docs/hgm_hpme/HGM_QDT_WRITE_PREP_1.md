# HGM-QDT-WRITE-PREP-1 — Read-Only Proposal Materialization Contract

## Purpose

HGM-QDT-WRITE-PREP-1 prepares the finalized HGM v0.1 bridge for a future QDT/WM write stage without executing writes. It translates the audit blockers from HGM-QDT-AUDIT-1 into concrete, typed contracts:

- bounded tensor proposal previews for future `SystemWriteProposal.content`
- HGM slot to WM local/canonical slot ID mapping
- q-spin placeholder to QH code-schema conversion previews
- rollback snapshot handshake requirements
- signature-level probes of QDT/WM write surfaces

## Safety Position

This stage is read-only. It does not instantiate live `SystemWriteProposal` writes, does not stage proposals in a `SystemCommitGate`, does not mutate `SharedSlotStore`, does not create `QHStorageRecord` instances in storage, and does not bind to live rollback snapshots.

## Added Modules

- `qdt_write_contract_probe.py`
- `qdt_proposal_materialization.py`
- `qdt_slot_mapping_plan.py`
- `qdt_qspin_qh_contract.py`
- `qdt_rollback_handshake.py`
- `hgm_qdt_write_prep_result.py`
- `hgm_qdt_write_prep_pipeline.py`

## Contract Outputs

### Proposal Materialization Contract

`build_proposal_materialization_contract(...)` converts HGM bridge payloads into `TensorProposalPreview` records. The preview vector is finite, bounded, deterministic, and shape-compatible with a future `SystemWriteProposal.content` vector, but it remains a tuple of floats rather than a live write tensor.

### Slot-ID Mapping Plan

`build_slot_id_mapping_plan(...)` maps `hgm_slot_*` style hook targets into WM-safe local slot IDs and deterministic `css-*` canonical slot previews using the WM canonical ID function when available.

### Q-Spin/QH Conversion Contract

`build_qspin_qh_conversion_contract(...)` maps HGM depth and geometry into QDT/QH-compatible fields and produces QH schema previews plus `qhrec-*` record ID previews. Placeholder q-spin IDs remain warnings until a real QSpinSignature/QH conversion stage exists.

### Rollback Snapshot Handshake

`build_rollback_snapshot_handshake(...)` produces rollback requirements against `SystemCommitGate.rollback_stack`, but marks them unbound because this read-only stage does not capture live WM rollback snapshots.

### QDT/WM Contract Probe

`probe_qdt_wm_write_contracts(...)` inspects the expected QDT/WM symbols and signatures without staging writes.

## High-Level Entry Point

Use:

```python
from mnemonic_cortex.hypergraph_manifold import build_hgm_qdt_write_prep_contracts

result = build_hgm_qdt_write_prep_contracts(trace_safe_memory_plan)
```

The returned `HGMQDTWritePrepResult` contains the probe, proposal materialization contract, slot mapping plan, QH conversion contract, rollback handshake, validation result, and traces.

## Known Limits

- No live writes.
- No actual `SystemWriteProposal` creation for commit.
- No slot-store mutation.
- No QH storage mutation.
- No live rollback snapshot capture.
- Placeholder q-spin IDs remain placeholders.

## Next Stage

`DEV-FLOW RUN HGM-QDT-WRITE-PREP-2 — Dry-Run SystemWriteProposal Builder, CommitGate Preflight Harness, and Non-Mutating End-to-End Write Simulation`
