import pytest

from mnemonic_cortex.reasoning_depth import PlannerRemediationGuidanceConfig, PlannerRemediationGuidanceError


def test_reason3b_remediation_blocks_auto_patch_config():
    with pytest.raises(PlannerRemediationGuidanceError):
        PlannerRemediationGuidanceConfig(enabled=True, allow_patch_application=True).validate()
