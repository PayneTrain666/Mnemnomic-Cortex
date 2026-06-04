"""High-level HGM-9 runtime integration evaluation entry point."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .hgm9_result import HGM9ReadinessOptions, HGM9RuntimeIntegrationResult, trace_hgm9
from .production_readiness_gate import score_production_readiness
from .qdt_runtime_evaluation import coerce_hgm9_options, evaluate_qdt_runtime_integration
from .slot_lattice_replay_benchmark import benchmark_slot_lattice_replay
from .validation import ValidationResult


def build_hgm9_runtime_integration_evaluation(records_or_result: Any = None, config=None, options: Optional[HGM9ReadinessOptions | Mapping[str, Any]] = None) -> HGM9RuntimeIntegrationResult:
    """Build HGM-9 runtime-readiness, replay, and production-gate results."""

    opts = coerce_hgm9_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    qdt_eval = evaluate_qdt_runtime_integration(records_or_result, config=config, options=opts)
    slot_replay = benchmark_slot_lattice_replay(records_or_result, config=config, options=opts)
    gate = score_production_readiness(records_or_result, qdt_evaluation=qdt_eval, slot_replay=slot_replay, config=config, options=opts)
    validation.merge(qdt_eval.validation).merge(slot_replay.validation)
    traces.extend(qdt_eval.trace_records + slot_replay.trace_records)
    trace = trace_hgm9("hgm9_pipeline.build_hgm9_runtime_integration_evaluation", validation, {"readiness_score": gate.score, "ready": gate.ready})
    traces.append(trace)
    return HGM9RuntimeIntegrationResult(
        qdt_runtime_evaluation=qdt_eval,
        slot_lattice_replay_benchmark=slot_replay,
        production_readiness_gate=gate,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"evaluation_first": True, "live_qdt_write": False, "production_enabled": gate.production_enabled},
    )
