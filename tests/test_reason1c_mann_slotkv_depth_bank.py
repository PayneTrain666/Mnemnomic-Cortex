import torch

from mnemonic_cortex.reasoning_depth import MANNSlotKVDepthBank, MANNSlotKVDepthBankConfig


def test_reason1c_mann_slotkv_depth_bank_shapes():
    bank = MANNSlotKVDepthBank(MANNSlotKVDepthBankConfig.enabled_default(key_dim=16, value_dim=20, slot_count=24))
    assert bank.keys.shape == (24, 8, 16)
    assert bank.values.shape == (24, 8, 20)
    metrics = bank.capacity_metrics()
    assert metrics["effective_subslots"] == 24 * 8
    assert metrics["theoretical_capacity_multiplier"] == 8
