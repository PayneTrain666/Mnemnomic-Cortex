from pathlib import Path

import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.memory import MemoryReadRequest, SlotWriteRequest


def test_checkpoint_roundtrip_with_shared_memory_and_hg_ltm(tmp_path: Path):
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_shared_memory_subsystem(num_slots=64, num_systems=8, device=torch.device("cpu"), dtype=torch.float32)
    model.enable_hg_episodic_ltm(write_mode="mirror")

    model.memory_write(
        SlotWriteRequest(
            requester_system="hg_ep_ltm",
            candidate_value_shape=[1, 16],
            requested_state="provisional",
            requested_memory_type="episodic",
        ),
        torch.randn(1, 16),
    )
    model.store_episodic_trace(
        episode_id="ep-1",
        episode_vectors=torch.randn(4, 16),
        step_range=(0, 3),
        tags=["checkpoint"],
    )

    ckpt = tmp_path / "cortex_ckpt.pt"
    model.save_checkpoint(str(ckpt))

    loaded = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    loaded.load_checkpoint(str(ckpt), strict=False)

    assert loaded.shared_memory_subsystem is not None
    assert loaded.hg_episodic_ltm is not None
    assert loaded.episodic_write_mode in {"legacy", "mirror", "shared_only"}
    read_out = loaded.memory_read(
        MemoryReadRequest(
            requester_system="hg_ep_ltm",
            query=torch.randn(1, 16),
            top_k=4,
            memory_type="episodic",
        )
    )
    assert read_out.values.shape[0] == 1


def test_checkpoint_roundtrip_preserves_ltm_slot_shapes(tmp_path: Path):
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=48,
        ltm_cgmn_slots=40,
        ltm_curved_slots=20,
        ltm_spatial_slots=24,
        ltm_enable_spatial_ltm=True,
    )
    ckpt = tmp_path / "cortex_capacity_ckpt.pt"
    model.save_checkpoint(str(ckpt))

    loaded = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=48,
        ltm_cgmn_slots=40,
        ltm_curved_slots=20,
        ltm_spatial_slots=24,
        ltm_enable_spatial_ltm=True,
    )
    loaded.load_checkpoint(str(ckpt), strict=False)

    assert loaded.long_term_memory.hg.q_memory_slots.shape[0] == 48
    assert loaded.long_term_memory.cgmn.memory_slots.shape[0] == 40
    assert loaded.long_term_memory.curved.memory_slots.shape[0] == 20
    assert loaded.long_term_memory.spatial_ltm is not None
    assert loaded.long_term_memory.spatial_ltm.slots == 24
