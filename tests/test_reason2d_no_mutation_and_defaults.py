import pytest
import torch

from mnemonic_cortex.reasoning_depth import ReasoningControllerAPI, ReasoningControllerAPIConfig, ReasoningControllerAPIError


def test_reason2d_public_api_rejects_write_permission_and_preserves_tensor():
    api = ReasoningControllerAPI(ReasoningControllerAPIConfig.enabled_default(key_dim=8, value_dim=8, slot_count=4))
    x = torch.randn(1, 2, 8)
    before = x.clone()
    with pytest.raises(ReasoningControllerAPIError):
        api.run_reasoning_pass(x, write_permission=True)
    assert torch.equal(x, before)


def test_reason2d_config_disallows_mutation_default_false():
    with pytest.raises(ReasoningControllerAPIError):
        ReasoningControllerAPIConfig(no_mutation_by_default=False).validate()
