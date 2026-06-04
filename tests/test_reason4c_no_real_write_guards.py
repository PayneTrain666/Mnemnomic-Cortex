import pytest

from mnemonic_cortex.reasoning_depth import (
    PersistenceLineClosureConfig,
    PersistenceLineClosureError,
    FinalSafetyAuditConfig,
    FinalSafetyAuditError,
)


def test_reason4c_no_real_write_guards():
    with pytest.raises(PersistenceLineClosureError):
        PersistenceLineClosureConfig(enabled=True, require_no_real_writes=False).validate()
    with pytest.raises(FinalSafetyAuditError):
        FinalSafetyAuditConfig(enabled=True, require_no_real_writes=False).validate()
