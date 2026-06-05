import torch

from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig


def test_reason1d_qh_metadata_is_present_and_not_fake_quantum_hardware():
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=12))
    result = adapter.propose_consolidation(
        canonical_slot_id="concept.qh.1",
        content="qh metadata content",
        bank_name="procedural_spcp",
        slot_index=2,
        depth_index=6,
        key=torch.randn(16),
        value=torch.randn(16),
    )

    qh = result["bank_proposal"]["qh_code"]
    assert qh["bank_code"] == "LTM_PROCEDURAL_SPCP"
    assert qh["depth_code"] == "Z6"
    assert qh["fake_quantum_hardware_claim"] is False
