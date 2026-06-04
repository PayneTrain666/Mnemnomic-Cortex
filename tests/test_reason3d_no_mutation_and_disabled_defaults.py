from mnemonic_cortex.reasoning_depth import (
    ReasoningReleaseCandidateConfig,
    ReasoningAPIFreezeConfig,
    ReasoningRegressionClosureConfig,
)


def test_reason3d_disabled_defaults_and_no_mutation():
    assert ReasoningReleaseCandidateConfig.disabled().enabled is False
    assert ReasoningAPIFreezeConfig.disabled().enabled is False
    assert ReasoningRegressionClosureConfig.disabled().enabled is False
    assert ReasoningReleaseCandidateConfig.enabled_default().no_mutation_by_default is True
    assert ReasoningAPIFreezeConfig.enabled_default().no_mutation_by_default is True
    assert ReasoningRegressionClosureConfig.enabled_default().no_mutation_by_default is True
