"""Slot-lattice replay benchmark for HGM-9.

The benchmark replays shared slot-lattice hook contracts as deterministic
read-only records. It never mutates the lattice or QDT/WM runtime memory.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .hgm4_result import SharedSlotLatticeHook, TraceSafeMemoryPlan, HGM4BridgeResult
from .hgm9_result import HGM9ReadinessOptions, SlotLatticeReplayBenchmarkResult, SlotLatticeReplayRecord, hgm9_stable_hash, trace_hgm9
from .qdt_runtime_evaluation import coerce_hgm9_options
from .validation import ValidationResult


def _extract_hooks(records_or_plan: Any) -> Tuple[SharedSlotLatticeHook, ...]:
    if isinstance(records_or_plan, SharedSlotLatticeHook):
        return (records_or_plan,)
    if isinstance(records_or_plan, TraceSafeMemoryPlan):
        return tuple(records_or_plan.slot_hooks or tuple())
    if isinstance(records_or_plan, HGM4BridgeResult):
        return tuple(records_or_plan.memory_plan.slot_hooks or tuple())
    if isinstance(records_or_plan, (list, tuple)):
        hooks: List[SharedSlotLatticeHook] = []
        for item in records_or_plan:
            hooks.extend(_extract_hooks(item))
        return tuple(hooks)
    return tuple()


def _score_hook(hook: SharedSlotLatticeHook) -> tuple[float, bool, str]:
    score = 1.0
    reasons = []
    replayable = True
    if not hook.hook_id or not hook.source_record_id or not hook.target_slot_id:
        score -= 0.45
        replayable = False
        reasons.append("missing stable hook/source/target id")
    if not hook.dry_run:
        score -= 0.35
        replayable = False
        reasons.append("hook is not dry-run")
    if hook.write_intent:
        score -= 0.25
        reasons.append("hook has write intent; replay remains read-only")
    if not hook.qspin_signature_id:
        score -= 0.10
        reasons.append("missing q-spin signature")
    if hook.confidence < 0.0 or hook.confidence > 1.0:
        score -= 0.20
        replayable = False
        reasons.append("confidence outside range")
    return max(0.0, min(1.0, score)), bool(replayable), "; ".join(reasons) or "hook is replayable as read-only contract"


def benchmark_slot_lattice_replay(records_or_plan: Any, config=None, options: Optional[HGM9ReadinessOptions | Mapping[str, Any]] = None) -> SlotLatticeReplayBenchmarkResult:
    """Benchmark slot-lattice hooks as deterministic replay contracts."""

    opts = coerce_hgm9_options(options)
    validation = ValidationResult()
    traces: List[Any] = []
    hooks = tuple(sorted(_extract_hooks(records_or_plan), key=lambda h: (h.target_slot_id, h.source_record_id, h.hook_id)))
    if not hooks:
        validation.warning("hgm9_slot_replay.empty", "no slot-lattice hooks supplied for replay benchmark", "hooks")
    if len(hooks) > opts.max_replay_hooks:
        validation.warning("hgm9_slot_replay.bounded", "hook count exceeded max_replay_hooks; truncated", "hooks")
    records: List[SlotLatticeReplayRecord] = []
    for hook in hooks[: opts.max_replay_hooks]:
        score, replayable, reason = _score_hook(hook)
        if not replayable:
            validation.warning("hgm9_slot_replay.hook_not_replayable", reason, hook.hook_id)
        trace = trace_hgm9("slot_lattice_replay_benchmark.record", validation, {"hook_id": hook.hook_id, "score": score, "metadata": hook.metadata})
        traces.append(trace)
        records.append(SlotLatticeReplayRecord(
            replay_id=f"hgm9_slot_replay_{hgm9_stable_hash(hook.hook_id, hook.target_slot_id, score)}",
            hook_id=hook.hook_id,
            source_record_id=hook.source_record_id,
            target_slot_id=hook.target_slot_id,
            depth_layer=hook.depth_layer.name,
            geometry_type=hook.geometry_type.value,
            replayable=replayable,
            score=score,
            reason=reason,
            trace_id=trace.trace_id,
            metadata={"dry_run": hook.dry_run, "write_intent": hook.write_intent, "live_qdt_write": False},
        ))
    aggregate = sum(record.score for record in records) / len(records) if records else 0.0
    replay_safe = bool(records) and all(record.replayable for record in records)
    trace = trace_hgm9("slot_lattice_replay_benchmark.benchmark_slot_lattice_replay", validation, {"record_count": len(records), "aggregate_score": aggregate, "replay_safe": replay_safe})
    traces.append(trace)
    return SlotLatticeReplayBenchmarkResult(
        records=tuple(records),
        aggregate_score=aggregate,
        replay_safe=replay_safe,
        validation=validation,
        trace_records=tuple(traces),
        metadata={"record_count": len(records), "live_qdt_write": False, "read_only_replay": True},
    )
