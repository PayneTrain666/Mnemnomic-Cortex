import pytest

from mnemonic_cortex.reasoning_depth import (
    BackendAuthorizationConfig,
    BackendAuthorizationError,
    BackendImplementationPlanConfig,
    BackendImplementationPlanError,
)


def test_future_backend_no_write_capable_code_guards():
    with pytest.raises(BackendAuthorizationError):
        BackendAuthorizationConfig(enabled=True, explicit_user_authorization=True, authorize_real_writes=True).validate()
    with pytest.raises(BackendImplementationPlanError):
        BackendImplementationPlanConfig(enabled=True, allow_write_capable_code=True).validate()
