import pytest
from mnemonic_cortex.reasoning_depth import DepthLatticeConfig, DepthLatticeConfigError, BankKind

def test_reason1a_config_defaults_to_eight_depths_and_capacity_multiplier():
    cfg = DepthLatticeConfig(slot_count=16, key_dim=32, value_dim=48)
    assert cfg.num_depths == 8
    assert cfg.effective_subslots == 128
    assert cfg.theoretical_capacity_multiplier == 8
    assert len(cfg.depth_roles) == 8

def test_reason1a_config_rejects_noncanonical_depth_count():
    with pytest.raises(DepthLatticeConfigError): DepthLatticeConfig(num_depths=7)

def test_reason1a_named_factory_presets():
    assert DepthLatticeConfig.wm().bank_kind == BankKind.WM
    assert DepthLatticeConfig.mann().bank_kind == BankKind.MANN
    assert DepthLatticeConfig.ltm().bank_kind == BankKind.LTM
