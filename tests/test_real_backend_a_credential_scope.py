import json
import pytest

from mnemonic_cortex.reasoning_depth import CredentialScopeModel, CredentialScopeConfig, CredentialScopeError


def test_real_backend_a_credential_scope_no_real_secret_loading():
    report = CredentialScopeModel(CredentialScopeConfig.enabled_test_injected()).build().to_dict()

    assert report["enabled"] is True
    assert report["real_secret_loaded"] is False
    assert "real_secret_loading" in report["blocked_scopes"]
    json.dumps(report)

    with pytest.raises(CredentialScopeError):
        CredentialScopeConfig(enabled=True, allow_real_secret_loading=True).validate()
