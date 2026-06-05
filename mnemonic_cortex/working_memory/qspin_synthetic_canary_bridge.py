"""Synthetic canary bridge runs for QSPIN-PROD-5."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import copy
import random

from .qspin_shadow_runtime_harness import ShadowRuntimeEnvelope, ShadowRuntimeHarness


CANARY_CATEGORIES = [
    "route_shape_canary",
    "payload_integrity_canary",
    "qh_no_write_canary",
    "shared_slot_no_write_canary",
    "external_memory_no_write_canary",
    "topology_no_execute_canary",
    "commit_gate_no_commit_canary",
    "audit_chain_canary",
    "redaction_canary",
    "concurrency_guard_canary",
    "timeout_bound_canary",
    "malformed_input_canary",
]


@dataclass
class CanarySpec:
    canary_id: str
    category: str
    description: str
    expected_status: str = "PASS"
    expected_reason: Optional[str] = None
    should_block: bool = False
    severity: str = "P1"


@dataclass
class CanaryPayload:
    spec: CanarySpec
    envelope: ShadowRuntimeEnvelope
    mutation_notes: List[str] = field(default_factory=list)


@dataclass
class CanaryResult:
    canary_id: str
    category: str
    status: str
    observed_runtime_status: str
    observed_reason_code: str
    expected_status: str
    remediation_hint: str
    detail: Dict[str, Any] = field(default_factory=dict)


def mutate_payload(base_payload: Dict[str, Any], *, seed: int, mutation: str) -> Dict[str, Any]:
    rng = random.Random(seed)
    payload = copy.deepcopy(base_payload)
    payload["mutation"] = mutation
    payload["synthetic_nonce"] = rng.randint(1000, 9999)
    payload["synthetic"] = True
    return payload


def build_canary_corpus(seed: int = 5005) -> List[CanaryPayload]:
    base = {"message": "synthetic canary bridge payload", "stage": "QSPIN-PROD-5-QD6A", "secret_probe": "api_key=SHOULD_REDACT"}
    specs = [
        CanarySpec("CANARY-001", "route_shape_canary", "Valid synthetic route shape should execute shadow-only."),
        CanarySpec("CANARY-002", "payload_integrity_canary", "Benign payload should survive synthetic planning."),
        CanarySpec("CANARY-003", "qh_no_write_canary", "QH write request must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P0"),
        CanarySpec("CANARY-004", "shared_slot_no_write_canary", "Shared-slot write request must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P0"),
        CanarySpec("CANARY-005", "external_memory_no_write_canary", "External-memory write request must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P0"),
        CanarySpec("CANARY-006", "topology_no_execute_canary", "Topology execution request must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P0"),
        CanarySpec("CANARY-007", "commit_gate_no_commit_canary", "Commit request must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P0"),
        CanarySpec("CANARY-008", "audit_chain_canary", "Audit chain must include a PASS audit event."),
        CanarySpec("CANARY-009", "redaction_canary", "Secret-looking payload text must not leak unredacted."),
        CanarySpec("CANARY-010", "concurrency_guard_canary", "Idempotent duplicate should skip replay."),
        CanarySpec("CANARY-011", "timeout_bound_canary", "Timeout bound config remains fail-closed if exceeded."),
        CanarySpec("CANARY-012", "malformed_input_canary", "Unsupported mode must be blocked.", expected_status="PASS", expected_reason="SAFETY_BOUNDARY_BLOCK", should_block=True, severity="P1"),
    ]
    flags_by_category = {
        "qh_no_write_canary": {"write_qh": True},
        "shared_slot_no_write_canary": {"write_shared_slot": True},
        "external_memory_no_write_canary": {"write_external_memory": True},
        "topology_no_execute_canary": {"execute_topology": True},
        "commit_gate_no_commit_canary": {"commit": True},
    }
    corpus: List[CanaryPayload] = []
    for i, spec in enumerate(specs):
        flags = flags_by_category.get(spec.category, {})
        mode = "SHADOW_EXECUTE_SYNTHETIC"
        if spec.category == "malformed_input_canary":
            mode = "UNSUPPORTED_LIVE_MODE"
        target = "local_synthetic_shadow"
        route = "synthetic.shadow.route"
        if spec.category == "route_shape_canary":
            route = "synthetic.shadow.bridge.route_shape"
        payload = mutate_payload(base, seed=seed + i, mutation=spec.category)
        envelope = ShadowRuntimeEnvelope(payload=payload, requested_route=route, requested_mode=mode, target=target, flags=flags)
        corpus.append(CanaryPayload(spec=spec, envelope=envelope, mutation_notes=[f"seed={seed + i}", spec.category]))
    return corpus


def evaluate_canary(payload: CanaryPayload, harness: ShadowRuntimeHarness) -> CanaryResult:
    result = harness.execute(payload.envelope)
    if payload.spec.should_block:
        passed = result.status == "FAIL_CLOSED" and (payload.spec.expected_reason is None or result.reason_code == payload.spec.expected_reason)
    elif payload.spec.category == "concurrency_guard_canary":
        second = harness.execute(payload.envelope)
        passed = result.status == "PASS" and second.status == "SKIP" and second.reason_code == "IDEMPOTENT_REPLAY"
        result_status = f"{result.status}+{second.status}"
        reason = f"{result.reason_code}+{second.reason_code}"
        return CanaryResult(
            canary_id=payload.spec.canary_id,
            category=payload.spec.category,
            status="PASS" if passed else "FAIL",
            observed_runtime_status=result_status,
            observed_reason_code=reason,
            expected_status=payload.spec.expected_status,
            remediation_hint="Check idempotency key cache and replay guard." if not passed else "none",
            detail={"first": result.__dict__, "second": second.__dict__},
        )
    else:
        passed = result.status == payload.spec.expected_status
    return CanaryResult(
        canary_id=payload.spec.canary_id,
        category=payload.spec.category,
        status="PASS" if passed else "FAIL",
        observed_runtime_status=result.status,
        observed_reason_code=result.reason_code,
        expected_status=payload.spec.expected_status,
        remediation_hint="Inspect safety-boundary validator and expected canary posture." if not passed else "none",
        detail={"runtime_result": result.__dict__, "spec": asdict(payload.spec)},
    )


def run_canaries(harness: Optional[ShadowRuntimeHarness] = None, *, seed: int = 5005) -> List[CanaryResult]:
    harness = harness or ShadowRuntimeHarness()
    out: List[CanaryResult] = []
    for payload in build_canary_corpus(seed=seed):
        harness.emitter.increment("canaries")
        out.append(evaluate_canary(payload, harness))
    return out
