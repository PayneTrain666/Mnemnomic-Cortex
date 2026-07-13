"""Regression: LTM qh_banks must migrate with model.to(cuda)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mnemonic_cortex.quantum_holographic import QuantumHologramConfig, QuantumHologramSlotBank
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


def _codebook_devices(bank: QuantumHologramSlotBank):
    devices = set()
    for attr in ("depth_codes", "bank_codes", "triplet_codes", "slot_codes"):
        table = getattr(bank.codebook, attr)
        for tensor in table.values():
            devices.add(tensor.device.type)
    return devices


def _tiny_ltm(**kwargs):
    defaults = dict(
        input_dim=32,
        output_dim=32,
        hg_slots=64,
        cgmn_slots=64,
        curved_slots=64,
        enable_spatial_ltm=False,
        enable_procedural_spcp=False,
        n_transformer_layers=1,
        hg_transformer_layers=0,
        cgmn_transformer_layers=0,
        curved_transformer_layers=0,
        fusion_transformer_layers=0,
        cross_model_attention_layers=0,
    )
    defaults.update(kwargs)
    return EnhancedTripleHybridMemory(**defaults)


def test_qh_banks_is_module_dict():
    mem = _tiny_ltm()
    assert isinstance(mem.qh_banks, nn.ModuleDict)
    assert "hg" in mem.qh_banks
    # Banks must be registered children so Module.to() reaches them.
    named = dict(mem.named_modules())
    assert "qh_banks.hg" in named
    assert "qh_banks.cgmn" in named
    assert "qh_banks.curved" in named


def test_slot_bank_to_moves_buffers_and_codebook():
    bank = QuantumHologramSlotBank(QuantumHologramConfig(hrr_dim=32, num_slots=8, bank_name="unit"))
    assert bank.holograms.device.type == "cpu"
    assert _codebook_devices(bank) == {"cpu"}

    if not torch.cuda.is_available():
        # Still verify CPU round-trip path for codebook migration.
        bank.to("cpu")
        assert bank.holograms.device.type == "cpu"
        assert _codebook_devices(bank) == {"cpu"}
        return

    bank.to("cuda")
    assert bank.holograms.device.type == "cuda"
    assert bank.triplet_counts.device.type == "cuda"
    assert _codebook_devices(bank) == {"cuda"}

    bank.to("cpu")
    assert bank.holograms.device.type == "cpu"
    assert _codebook_devices(bank) == {"cpu"}


def test_triple_hybrid_to_cuda_migrates_qh_banks():
    if not torch.cuda.is_available():
        return

    mem = _tiny_ltm(enable_spatial_ltm=True, spatial_slots=64)
    mem.to("cuda")

    for name, bank in mem.qh_banks.items():
        assert bank.holograms.device.type == "cuda", name
        assert _codebook_devices(bank) == {"cuda"}, name

    # store_batch must not raise device mismatch after model.to(cuda)
    slots = torch.tensor([0, 1], device="cuda")
    vec = torch.randn(2, 32, device="cuda")
    stats = mem.qh_banks["hg"].store_batch(
        slot_indices=slots,
        anchor=vec,
        direction=vec,
        phase=vec,
        depth_index=0,
    )
    assert stats["stored"] >= 1.0