import pytest

from mnemonic_cortex.reasoning_depth import (
    PersistenceBackendConfig,
    PersistenceBackendError,
    PersistenceRecoveryConfig,
    PersistenceRecoveryError,
)


def test_reason4b_no_real_write_guards():
    with pytest.raises(PersistenceBackendError):
        PersistenceBackendConfig(enabled=True, allow_real_writes=True).validate()
    with pytest.raises(PersistenceRecoveryError):
        PersistenceRecoveryConfig(enabled=True, allow_real_rollback=True).validate()
