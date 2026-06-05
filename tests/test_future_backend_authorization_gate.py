import json

from mnemonic_cortex.reasoning_depth import (
    BackendAuthorizationGate,
    BackendAuthorizationConfig,
    BackendStoreTarget,
)


def test_future_backend_authorization_gate_planning_only():
    gate = BackendAuthorizationGate(BackendAuthorizationConfig.planning_authorized(BackendStoreTarget.SQLITE))
    decision = gate.decide(lineage={"stage": "FUTURE-BACKEND-AUTHORIZATION"}).to_dict()

    assert decision["status"] == "plan_only_authorized"
    assert decision["real_store_write_authorized"] is False
    assert decision["write_capable_code_authorized"] is False
    assert "write_capable_backend_code" in decision["blocked_actions"]
    json.dumps(decision)
