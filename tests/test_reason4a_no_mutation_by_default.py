import pytest

from mnemonic_cortex.reasoning_depth import (
    ReasoningPersistenceAdapterConfig,
    ReasoningCommitInterfaceConfig,
    StoreSafetyContractConfig,
    ReasoningPersistenceAdapterError,
    ReasoningCommitInterfaceError,
    StoreSafetyError,
)


def test_reason4a_no_mutation_by_default_config_guards():
    with pytest.raises(ReasoningPersistenceAdapterError):
        ReasoningPersistenceAdapterConfig(enabled=True, no_mutation_by_default=False).validate()
    with pytest.raises(ReasoningCommitInterfaceError):
        ReasoningCommitInterfaceConfig(enabled=True, require_write_permission=False).validate()
    with pytest.raises(StoreSafetyError):
        StoreSafetyContractConfig(enabled=True, no_mutation_by_default=False).validate()
