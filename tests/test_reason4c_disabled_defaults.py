from mnemonic_cortex.reasoning_depth import (
    PersistenceLineClosureConfig,
    ReasoningIntegrationIndexConfig,
    FinalSafetyAuditConfig,
)


def test_reason4c_disabled_defaults():
    assert PersistenceLineClosureConfig.disabled().enabled is False
    assert ReasoningIntegrationIndexConfig.disabled().enabled is False
    assert FinalSafetyAuditConfig.disabled().enabled is False
    assert PersistenceLineClosureConfig.enabled_default().no_mutation_by_default is True
    assert FinalSafetyAuditConfig.enabled_default().require_no_real_writes is True
