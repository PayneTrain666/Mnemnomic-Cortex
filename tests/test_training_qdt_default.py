from __future__ import annotations

import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def _tiny_qdt_cortex() -> EnhancedMnemonicCortex:
    return EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        working_memory_fabric="qdt",
        qdt_hardware_profile="compact",
        qdt_num_slots=4,
        qdt_transformer_layers=1,
    )


def test_qdt_mount_keeps_qspin_and_write_permissions_inert_by_default():
    model = _tiny_qdt_cortex()
    fabric = model.describe_working_memory_fabric()
    config = fabric["qdt_config"]

    assert fabric["fabric"] == "qdt"
    assert fabric["working_memory_class"] == "QDTWMCompatibilityWrapper"
    assert config["qspin_live_activation"] is False
    assert config["qspin_live_mode"] == "disabled"
    assert config["qspin_live_allow_routing"] is False
    assert config["qspin_live_allow_payload_transfer"] is False
    assert config["qspin_live_allow_shared_slot_write"] is False
    assert config["qspin_live_allow_qh_storage_write"] is False
    assert config["qspin_live_allow_commit_execution"] is False


def test_qdt_working_memory_receives_gradients_during_sequence_training():
    model = CortexSeqModel(
        vocab_size=16,
        d_model=16,
        cms_enabled=False,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_enable_spatial_ltm=False,
        ltm_auto_wire_spatial=False,
        working_memory_fabric="qdt",
        qdt_hardware_profile="compact",
        qdt_num_slots=4,
        qdt_transformer_layers=1,
        qdt_qspin_live_activation=False,
    )
    model.train()
    logits = model(torch.randint(0, 16, (2, 3)))
    logits.square().mean().backward()

    wrapper = model.cortex.working_memory
    qdt = getattr(wrapper, "qdt_working_memory", wrapper)
    assert any(parameter.grad is not None for parameter in qdt.parameters())

