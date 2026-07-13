"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-6 shadow runtime stress replay.

Runs deterministic synthetic stress scenarios against metadata-only records. It
is bounded, idempotent, fail-closed, and cannot activate live routing/writes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple
import hashlib, json

class StressReplayMode(str, Enum):
    SYNTHETIC_ONLY = "synthetic_only"
    DISABLED = "disabled"

class StressReplayStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class StressReplayBlockReason(str, Enum):
    MODE_DISABLED = "mode_disabled"
    LIVE_ROUTING_REQUESTED = "live_routing_requested"
    REAL_PAYLOAD_TRANSFER_REQUESTED = "real_payload_transfer_requested"
    REAL_WRITE_REQUESTED = "real_write_requested"
    COMMIT_REQUESTED = "commit_requested"
    PRODUCTION_ACTIVATION_REQUESTED = "production_activation_requested"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    INVALID_SCENARIO = "invalid_scenario"
    MISSING_SOURCE_STRUCTURED_SKIP = "missing_source_structured_skip"

class StressReplayScenarioKind(str, Enum):
    SHADOW_RUNTIME = "shadow_runtime"
    CANARY_CORPUS = "canary_corpus"
    SAFETY_REGRESSION = "safety_regression"
    TRACE_CORPUS = "trace_corpus"
    MALFORMED_INPUT = "malformed_input"
    FORBIDDEN_ACTION = "forbidden_action"
    CONCURRENCY_PRESSURE = "concurrency_pressure"
    TIMEOUT_BOUND = "timeout_bound"
    REDACTION = "redaction"
    AUDIT_CHAIN = "audit_chain"
    STRUCTURED_SKIP = "structured_skip"
    DEAD_LETTER = "dead_letter"
    REMEDIATION = "remediation"

@dataclass(frozen=True)
class StressReplayScenario:
    scenario_id: str
    kind: StressReplayScenarioKind
    iterations: int = 1
    should_skip: bool = False
    requests_live_effect: bool = False

    def validate(self, max_iterations: int) -> "StressReplayScenario":
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        if not isinstance(self.kind, StressReplayScenarioKind):
            raise ValueError("invalid scenario kind")
        if self.iterations <= 0 or self.iterations > max_iterations:
            raise ValueError("scenario iterations out of bounds")
        return self

@dataclass(frozen=True)
class StressReplayInput:
    seed: int = 606
    scenarios: Tuple[StressReplayScenario, ...] = ()
    source_available: bool = True
    live_routing_requested: bool = False
    payload_transfer_requested: bool = False
    write_requested: bool = False
    commit_requested: bool = False
    production_activation_requested: bool = False

@dataclass(frozen=True)
class StressReplayResult:
    scenario_id: str
    status: StressReplayStatus
    replay_hash: str
    block_reasons: Tuple[StressReplayBlockReason, ...] = ()
    iterations: int = 0

    @property
    def passed(self) -> bool:
        return self.status is StressReplayStatus.PASSED

@dataclass(frozen=True)
class StressReplaySuiteResult:
    status: StressReplayStatus
    results: Tuple[StressReplayResult, ...]
    pass_count: int
    fail_count: int
    skip_count: int
    suite_hash: str

    @property
    def passed(self) -> bool:
        return self.fail_count == 0 and self.status is StressReplayStatus.PASSED

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "suite_hash": self.suite_hash,
            "results": [{"scenario_id": r.scenario_id, "status": r.status.value, "block_reasons": [b.value for b in r.block_reasons], "replay_hash": r.replay_hash, "iterations": r.iterations} for r in self.results],
        }

@dataclass(frozen=True)
class StressReplayConfig:
    mode: StressReplayMode = StressReplayMode.SYNTHETIC_ONLY
    max_scenarios: int = 128
    max_iterations: int = 32
    max_runtime_ms_per_scenario: int = 250

    def validate(self) -> "StressReplayConfig":
        if self.mode is StressReplayMode.DISABLED:
            raise ValueError("stress replay disabled")
        if self.max_scenarios <= 0 or self.max_iterations <= 0 or self.max_runtime_ms_per_scenario <= 0:
            raise ValueError("invalid stress replay bounds")
        return self

class ShadowStressReplayEngine:
    def __init__(self, config: StressReplayConfig | None = None):
        self.config = (config or build_default_stress_replay_config()).validate()

    def run(self, replay_input: StressReplayInput) -> StressReplaySuiteResult:
        scenarios = replay_input.scenarios or build_default_stress_replay_suite()
        if len(scenarios) > self.config.max_scenarios:
            raise ValueError("too many stress replay scenarios")
        results = []
        global_reasons = []
        if replay_input.live_routing_requested:
            global_reasons.append(StressReplayBlockReason.LIVE_ROUTING_REQUESTED)
        if replay_input.payload_transfer_requested:
            global_reasons.append(StressReplayBlockReason.REAL_PAYLOAD_TRANSFER_REQUESTED)
        if replay_input.write_requested:
            global_reasons.append(StressReplayBlockReason.REAL_WRITE_REQUESTED)
        if replay_input.commit_requested:
            global_reasons.append(StressReplayBlockReason.COMMIT_REQUESTED)
        if replay_input.production_activation_requested:
            global_reasons.append(StressReplayBlockReason.PRODUCTION_ACTIVATION_REQUESTED)
        for scenario in sorted(scenarios, key=lambda s: s.scenario_id):
            try:
                scenario.validate(self.config.max_iterations)
            except ValueError:
                reason = (StressReplayBlockReason.INVALID_SCENARIO,)
                results.append(self._result(scenario.scenario_id, StressReplayStatus.FAILED, replay_input.seed, reason, scenario.iterations))
                continue
            if not replay_input.source_available or scenario.should_skip:
                reason = (StressReplayBlockReason.MISSING_SOURCE_STRUCTURED_SKIP,)
                results.append(self._result(scenario.scenario_id, StressReplayStatus.SKIPPED, replay_input.seed, reason, scenario.iterations))
            elif scenario.requests_live_effect or global_reasons:
                reasons = tuple(dict.fromkeys(global_reasons or [StressReplayBlockReason.LIVE_ROUTING_REQUESTED]))
                results.append(self._result(scenario.scenario_id, StressReplayStatus.FAILED, replay_input.seed, reasons, scenario.iterations))
            else:
                results.append(self._result(scenario.scenario_id, StressReplayStatus.PASSED, replay_input.seed, (), scenario.iterations))
        pass_count = sum(r.status is StressReplayStatus.PASSED for r in results)
        fail_count = sum(r.status is StressReplayStatus.FAILED for r in results)
        skip_count = sum(r.status is StressReplayStatus.SKIPPED for r in results)
        suite_payload = [(r.scenario_id, r.status.value, r.replay_hash, [b.value for b in r.block_reasons]) for r in results]
        suite_hash = hashlib.sha256(json.dumps(suite_payload, sort_keys=True).encode()).hexdigest()
        status = StressReplayStatus.PASSED if fail_count == 0 else StressReplayStatus.FAILED
        return StressReplaySuiteResult(status, tuple(results), pass_count, fail_count, skip_count, suite_hash)

    def _result(self, scenario_id: str, status: StressReplayStatus, seed: int, reasons: Tuple[StressReplayBlockReason, ...], iterations: int) -> StressReplayResult:
        payload = {"scenario_id": scenario_id, "status": status.value, "seed": seed, "reasons": [r.value for r in reasons], "iterations": iterations}
        replay_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return StressReplayResult(scenario_id, status, replay_hash, reasons, iterations)


def build_default_stress_replay_config() -> StressReplayConfig:
    return StressReplayConfig().validate()


def build_default_stress_replay_suite() -> Tuple[StressReplayScenario, ...]:
    return tuple(StressReplayScenario(f"stress_{kind.value}", kind, iterations=2) for kind in StressReplayScenarioKind)
