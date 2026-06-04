import pytest

from mnemonic_cortex.reasoning_depth import ReasoningAPIFreezeConfig, ReasoningAPIFreezeError


def test_reason3d_api_freeze_blocks_breaking_changes():
    with pytest.raises(ReasoningAPIFreezeError):
        ReasoningAPIFreezeConfig(enabled=True, allow_api_breaking_changes=True).validate()
