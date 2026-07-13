"""
Plain-language summary
----------------------
What this file is for: QSPIN bridge contract, gate, sandbox, or observability helper.
How it fits in the system: Documents and guards a future optional bridge; not part of normal live memory routing today.
Status: INERT
Important notes for non-coders: Project policy keeps QSPIN disabled unless a later stage explicitly authorizes guarded activation.

Technical notes (original):
QSPIN-PROD-5 guarded live-adjacent shadow runtime harness.

This harness is deliberately synthetic-only. It mirrors production-shaped envelopes
but blocks every path that would perform live routing, topology execution, payload
transfer, commits, QH writes, shared-slot writes, external-memory writes, or
production activation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import threading
import time

from .qspin_runtime_observability import ObservabilityEmitter, redact_value

SHADOW_IDLE = "SHADOW_IDLE"
SHADOW_VALIDATE = "SHADOW_VALIDATE"
SHADOW_PLAN = "SHADOW_PLAN"
SHADOW_EXECUTE_SYNTHETIC = "SHADOW_EXECUTE_SYNTHETIC"
SHADOW_AUDIT = "SHADOW_AUDIT"
SHADOW_FAIL_CLOSED = "SHADOW_FAIL_CLOSED"

FORBIDDEN_FLAGS = {
    "live_route",
    "execute_topology",
    "transfer_payload",
    "write_working_memory",
    "write_long_term_memory",
    "write_qh",
    "write_shared_slot",
    "write_external_memory",
    "commit",
    "production_activation",
    "network_call",
    "mutate_production_config",
}

PRODUCTION_TARGET_MARKERS = ("prod", "production", "live", "external", "qh://", "memory://", "http://", "https://")


class FailClosedError(RuntimeError):
    """Raised internally when the shadow harness blocks unsafe execution."""


@dataclass
class ShadowRuntimeConfig:
    stage: str = "QSPIN-PROD-5-QD6A"
    dry_run: bool = True
    sandbox: bool = True
    shadow_only: bool = True
    allow_live_routing: bool = False
    allow_payload_transfer: bool = False
    allow_topology_execution: bool = False
    allow_qh_writes: bool = False
    allow_shared_slot_writes: bool = False
    allow_external_memory_writes: bool = False
    allow_commits: bool = False
    allow_network: bool = False
    allow_production_targets: bool = False
    max_payload_bytes: int = 65536
    max_steps: int = 8
    timeout_ms: int = 250
    lineage: Dict[str, str] = field(default_factory=lambda: {
        "qd6a": "qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip",
        "qspin8": "qspin8_qd6a_release_pack.zip",
        "prod0": "qspin_prod0_qd6a_release_pack.zip",
        "prod1": "qspin_prod1_qd6a_release_pack.zip",
        "prod2": "qspin_prod2_qd6a_release_pack.zip",
        "prod3": "qspin_prod3_qd6a_release_pack_refresh.zip",
        "prod4": "qspin_prod4_qd6a_release_pack.zip",
    })


@dataclass
class ShadowRuntimeEnvelope:
    payload: Dict[str, Any]
    requested_route: str = "synthetic.shadow.route"
    requested_mode: str = SHADOW_EXECUTE_SYNTHETIC
    target: str = "local_synthetic_shadow"
    flags: Dict[str, bool] = field(default_factory=dict)
    idempotency_key: Optional[str] = None
    source_stage: str = "QSPIN-PROD-5-QD6A"


@dataclass
class ShadowRuntimeResult:
    status: str
    mode: str
    reason_code: str
    idempotency_key: str
    simulated_route: Optional[str] = None
    planned_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def fingerprint(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, default=str)
        # Ignore elapsed jitter by hashing stable fields only.
        stable = {
            "status": self.status,
            "mode": self.mode,
            "reason_code": self.reason_code,
            "idempotency_key": self.idempotency_key,
            "simulated_route": self.simulated_route,
            "planned_actions": self.planned_actions,
            "blocked_actions": self.blocked_actions,
            "diagnostics": self.diagnostics,
        }
        return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


class ShadowRuntimeHarness:
    def __init__(self, config: Optional[ShadowRuntimeConfig] = None, emitter: Optional[ObservabilityEmitter] = None):
        self.config = config or ShadowRuntimeConfig()
        self.emitter = emitter or ObservabilityEmitter(stage=self.config.stage, lineage=self.config.lineage)
        self._lock = threading.RLock()
        self._seen: Dict[str, ShadowRuntimeResult] = {}

    def idempotency_key_for(self, envelope: ShadowRuntimeEnvelope) -> str:
        stable = {
            "payload": redact_value(envelope.payload),
            "requested_route": envelope.requested_route,
            "requested_mode": envelope.requested_mode,
            "target": envelope.target,
            "flags": envelope.flags,
            "source_stage": envelope.source_stage,
        }
        return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()

    def validate_safety_boundary(self, envelope: ShadowRuntimeEnvelope) -> Tuple[bool, str, List[str]]:
        cfg = self.config
        blocked: List[str] = []
        if not (cfg.dry_run and cfg.sandbox and cfg.shadow_only):
            blocked.append("config_not_dry_run_sandbox_shadow")
        if cfg.allow_live_routing or cfg.allow_payload_transfer or cfg.allow_topology_execution:
            blocked.append("config_allows_live_behavior")
        if cfg.allow_qh_writes or cfg.allow_shared_slot_writes or cfg.allow_external_memory_writes:
            blocked.append("config_allows_persistent_memory_writes")
        if cfg.allow_commits or cfg.allow_network or cfg.allow_production_targets:
            blocked.append("config_allows_external_or_production_side_effects")

        try:
            payload_bytes = len(json.dumps(envelope.payload, sort_keys=True, default=str).encode("utf-8"))
        except Exception:
            blocked.append("payload_not_json_serializable")
            payload_bytes = cfg.max_payload_bytes + 1
        if payload_bytes > cfg.max_payload_bytes:
            blocked.append("payload_too_large")

        target_lower = str(envelope.target).lower()
        route_lower = str(envelope.requested_route).lower()
        if any(marker in target_lower for marker in PRODUCTION_TARGET_MARKERS):
            blocked.append("production_target_blocked")
        if "live" in route_lower or "production" in route_lower or "prod" in route_lower:
            blocked.append("live_or_production_route_blocked")

        requested_forbidden = sorted(flag for flag, enabled in envelope.flags.items() if enabled and flag in FORBIDDEN_FLAGS)
        blocked.extend(f"forbidden_flag:{flag}" for flag in requested_forbidden)

        if envelope.requested_mode not in {SHADOW_VALIDATE, SHADOW_PLAN, SHADOW_EXECUTE_SYNTHETIC, SHADOW_AUDIT}:
            blocked.append("unsupported_shadow_mode")

        if blocked:
            return False, "SAFETY_BOUNDARY_BLOCK", blocked
        return True, "SAFETY_BOUNDARY_PASS", []

    def plan_synthetic_route(self, envelope: ShadowRuntimeEnvelope) -> List[str]:
        return [
            "validate_envelope",
            "redact_payload",
            "derive_synthetic_route_shape",
            "simulate_bridge_decision_no_payload_transfer",
            "emit_audit_only",
        ][: self.config.max_steps]

    def synthetic_route_simulator(self, envelope: ShadowRuntimeEnvelope) -> str:
        route_seed = f"{envelope.requested_route}|{envelope.target}|{envelope.source_stage}"
        return "synthetic://shadow/" + hashlib.sha256(route_seed.encode()).hexdigest()[:16]

    def dry_run_executor(self, envelope: ShadowRuntimeEnvelope) -> ShadowRuntimeResult:
        started = time.time()
        key = envelope.idempotency_key or self.idempotency_key_for(envelope)
        with self._lock:
            if key in self._seen:
                prior = self._seen[key]
                self.emitter.emit_audit("shadow_runtime_idempotent_replay", "SKIP", "IDEMPOTENT_REPLAY", detail={"idempotency_key": key})
                return ShadowRuntimeResult(
                    status="SKIP",
                    mode=SHADOW_AUDIT,
                    reason_code="IDEMPOTENT_REPLAY",
                    idempotency_key=key,
                    simulated_route=prior.simulated_route,
                    planned_actions=prior.planned_actions,
                    blocked_actions=[],
                    diagnostics={"replayed_from_seen_key": True, "prior_status": prior.status},
                    audit_events=[a for a in self.emitter.summary()["audits"][-2:]],
                    elapsed_ms=round((time.time() - started) * 1000.0, 3),
                )

            ok, reason, blocked = self.validate_safety_boundary(envelope)
            if not ok:
                self.emitter.emit_audit("shadow_runtime_block", "FAIL_CLOSED", reason, detail={"blocked_actions": blocked, "envelope": asdict(envelope)})
                result = ShadowRuntimeResult(
                    status="FAIL_CLOSED",
                    mode=SHADOW_FAIL_CLOSED,
                    reason_code=reason,
                    idempotency_key=key,
                    blocked_actions=blocked,
                    diagnostics={"synthetic_only": True, "side_effects": "none"},
                    audit_events=[a for a in self.emitter.summary()["audits"][-2:]],
                    elapsed_ms=round((time.time() - started) * 1000.0, 3),
                )
                self._seen[key] = result
                return result

            planned = self.plan_synthetic_route(envelope)
            simulated_route = self.synthetic_route_simulator(envelope)
            elapsed_ms = round((time.time() - started) * 1000.0, 3)
            if elapsed_ms > self.config.timeout_ms:
                self.emitter.emit_audit("shadow_runtime_timeout", "FAIL_CLOSED", "TIMEOUT_BOUND_EXCEEDED", detail={"elapsed_ms": elapsed_ms})
                result = ShadowRuntimeResult(
                    status="FAIL_CLOSED",
                    mode=SHADOW_FAIL_CLOSED,
                    reason_code="TIMEOUT_BOUND_EXCEEDED",
                    idempotency_key=key,
                    blocked_actions=["timeout_bound_exceeded"],
                    elapsed_ms=elapsed_ms,
                )
                self._seen[key] = result
                return result

            self.emitter.emit_audit("shadow_runtime_execute", "PASS", "SYNTHETIC_SHADOW_EXECUTED", detail={"simulated_route": simulated_route, "planned_actions": planned})
            result = ShadowRuntimeResult(
                status="PASS",
                mode=SHADOW_EXECUTE_SYNTHETIC,
                reason_code="SYNTHETIC_SHADOW_EXECUTED",
                idempotency_key=key,
                simulated_route=simulated_route,
                planned_actions=planned,
                diagnostics={
                    "synthetic": True,
                    "shadow": True,
                    "dry_run": True,
                    "sandbox": True,
                    "payload_transferred": False,
                    "persistent_writes": False,
                },
                audit_events=[a for a in self.emitter.summary()["audits"][-2:]],
                elapsed_ms=elapsed_ms,
            )
            self._seen[key] = result
            return result

    def execute(self, envelope: ShadowRuntimeEnvelope) -> ShadowRuntimeResult:
        started = time.time()
        try:
            return self.dry_run_executor(envelope)
        except Exception as exc:  # defensive fail-closed shell
            self.emitter.emit_audit("shadow_runtime_exception", "FAIL_CLOSED", "UNHANDLED_EXCEPTION_FAIL_CLOSED", detail={"error": repr(exc)})
            return ShadowRuntimeResult(
                status="FAIL_CLOSED",
                mode=SHADOW_FAIL_CLOSED,
                reason_code="UNHANDLED_EXCEPTION_FAIL_CLOSED",
                idempotency_key=envelope.idempotency_key or self.idempotency_key_for(envelope),
                blocked_actions=["unhandled_exception"],
                diagnostics={"error": repr(exc), "side_effects": "none"},
                audit_events=[a for a in self.emitter.summary()["audits"][-2:]],
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
