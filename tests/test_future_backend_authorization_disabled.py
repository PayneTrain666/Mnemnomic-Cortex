from mnemonic_cortex.reasoning_depth import BackendAuthorizationGate, BackendAuthorizationConfig


def test_future_backend_authorization_disabled_blocks_all():
    decision = BackendAuthorizationGate(BackendAuthorizationConfig.disabled()).decide().to_dict()

    assert decision["status"] == "not_authorized"
    assert "planning_not_authorized" in decision["blocked_actions"]
    assert decision["real_store_write_authorized"] is False
