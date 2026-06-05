from mnemonic_cortex.reasoning_depth import LTMDepthBanks


def test_reason1d_ltm_depth_banks_shapes():
    banks = LTMDepthBanks.enabled_default(key_dim=16, value_dim=20, slot_count=24)
    for name, bank in banks.all_banks().items():
        assert bank.keys.shape == (24, 8, 16)
        assert bank.values.shape == (24, 8, 20)
        metrics = bank.capacity_metrics()
        assert metrics["effective_subslots"] == 24 * 8
        assert metrics["theoretical_capacity_multiplier"] == 8
        assert metrics["bank_name"] == name
