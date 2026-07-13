"""
Plain-language summary
----------------------
What this file is for: Hypergraph / HGM manifold module: hgm qdt write prep3 pipeline.
How it fits in the system: Scaffolding for hypergraph probability / procedural manifold routing and write preparation.
Status: LOW-USE / SCAFFOLD (varies)
Important notes for non-coders: Many modules are stage artifacts or guarded write-prep rather than the default forward path.

Technical notes (original):
High-level HGM/QDT WRITE-PREP-3 pipeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .hgm_qdt_write_prep3_result import HGMQDTWritePrep3Options, HGMQDTWritePrep3Result
from .hgm_qdt_write_prep_result import trace_write_prep, write_prep_stable_hash
from .qdt_in_memory_commitgate_simulation import simulate_in_memory_commitgate
from .qdt_rollback_replay_verification import verify_rollback_replay
from .qdt_synthetic_slot_store import coerce_write_prep3_options
from .validation import ValidationResult


def build_hgm_qdt_write_prep_3(
    prep2_result_or_proposals: Any,
    config=None,
    options: Optional[HGMQDTWritePrep3Options | Mapping[str, Any]] = None,
) -> HGMQDTWritePrep3Result:
    """Build isolated CommitGate simulation and rollback replay verification.

    This function does not stage, commit, write shared slots, write QH storage,
    or mutate a real rollback stack. It only mutates synthetic immutable records
    by returning new dataclass instances.
    """
    opts = coerce_write_prep3_options(options)
    validation = ValidationResult()
    traces = []
    simulation = simulate_in_memory_commitgate(prep2_result_or_proposals, config=config, options=opts)
    validation.merge(simulation.validation)
    traces.extend(simulation.trace_records)
    rollback = verify_rollback_replay(simulation, config=config, options=opts) if opts.verify_rollback_after_simulation else None
    if rollback is None:
        validation.warning("prep3.rollback.verification_disabled", "rollback verification disabled by options", "options.verify_rollback_after_simulation")
        rollback = verify_rollback_replay(simulation, config=config, options=opts)
    validation.merge(rollback.validation)
    traces.extend(rollback.trace_records)
    final_trace = trace_write_prep("hgm_qdt_write_prep3_pipeline.build_hgm_qdt_write_prep_3", validation, {
        "simulation_id": simulation.simulation_id,
        "rollback_verified": rollback.verified,
        "live_qdt_wm_mutated": False,
        "stage_called": False,
        "commit_called": False,
    })
    traces.append(final_trace)
    return HGMQDTWritePrep3Result(
        commitgate_simulation=simulation,
        rollback_replay=rollback,
        validation=validation,
        trace_records=tuple(traces),
        metadata={
            "stage": "HGM-QDT-WRITE-PREP-3",
            "read_only": True,
            "isolated_in_memory": True,
            "synthetic_shared_slot_store": True,
            "live_write_executed": False,
            "system_commitgate_stage_called": False,
            "system_commitgate_commit_called": False,
            "shared_slot_store_mutated": False,
            "qh_storage_mutated": False,
            "rollback_stack_mutated": False,
            "result_id": f"hgm_qdt_write_prep_3_{write_prep_stable_hash(simulation.simulation_id, rollback.verification_id)}",
        },
    )
