from mnemonic_cortex.reasoning_depth import (
    reasoning_persistence_line_closure_contract,
    reasoning_final_safety_audit_contract,
    backend_authorization_contract,
    backend_threat_model_contract,
    backend_implementation_plan_contract,
)


def test_future_backend_reason4c_compatibility_and_contracts():
    assert reasoning_persistence_line_closure_contract()["stage"] == "REASON-4C"
    assert reasoning_final_safety_audit_contract()["future_backend_requires_explicit_authorization"] is True
    assert backend_authorization_contract()["stage"] == "FUTURE-BACKEND-AUTHORIZATION"
    assert backend_threat_model_contract()["real_store_write_authorized"] is False
    assert backend_implementation_plan_contract()["write_capable_code_generated"] is False
