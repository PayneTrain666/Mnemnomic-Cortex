from mnemonic_cortex.reasoning_depth import PersistenceLineClosure, PersistenceLineClosureConfig


def test_reason4c_final_closure_command_policy_blocks_hidden_backend():
    report = PersistenceLineClosure(PersistenceLineClosureConfig.enabled_default()).close().to_dict()
    blocked = report["decision_record"]["blocked_actions"]

    assert "hidden_backend_activation" in blocked
    assert "automatic_persistence_write" in blocked
    assert "separate_future_backend_authorization_only" in report["decision_record"]["allowed_next_actions"]
