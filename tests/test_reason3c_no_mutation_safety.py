import pytest
import torch

from mnemonic_cortex.reasoning_depth import (
    ControllerPlannerIntegration,
    ControllerPlannerIntegrationConfig,
    ControllerPlannerIntegrationError,
)


def test_reason3c_no_write_permission_and_no_mutation():
    integration = ControllerPlannerIntegration(ControllerPlannerIntegrationConfig.enabled_default())
    x = torch.randn(1, 2, 8)
    before = x.clone()

    with pytest.raises(ControllerPlannerIntegrationError):
        integration.run(x, content="blocked", write_permission=True)

    assert torch.equal(x, before)
