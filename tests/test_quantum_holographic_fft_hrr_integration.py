import torch

from mnemonic_cortex.consolidated_memory import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from mnemonic_cortex.memory_curved import EnhancedCurvedMemory
from mnemonic_cortex.quantum_holographic import (
    FFTHRRTripletStacker,
    QuantumHologramCodebook,
    QuantumHologramConfig,
)
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


def test_fft_hrr_triplet_codebook_stacks_with_low_crosstalk():
    cfg = QuantumHologramConfig(hrr_dim=128, num_slots=64, num_depths=8, bank_name="test")
    codebook = QuantumHologramCodebook(cfg, bank_names=["test", "aux"])
    stacker = FFTHRRTripletStacker(cfg, codebook)
    a1 = torch.randn(128)
    d1 = torch.randn(128)
    p1 = torch.randn(128)
    a2 = torch.randn(128)
    d2 = torch.randn(128)
    p2 = torch.randn(128)
    h1 = stacker.stack_triplet(anchor=a1, direction=d1, phase=p1, slot_index=3, depth_index=1, bank_name="test")
    h2 = stacker.stack_triplet(anchor=a2, direction=d2, phase=p2, slot_index=31, depth_index=6, bank_name="aux")
    inter = stacker.interference_score(h1, h2)
    assert inter < 0.95


def test_wm_and_ltm_qh_banks_store_triplets():
    torch.manual_seed(7)
    x = torch.randn(4, 6, 32)
    wm = EnhancedCurvedMemory(input_dim=32, hidden_dim=64, mem_slots=32, topk=4)
    wm(x, operation="write")
    wm_summary = wm.qh_slot_bank.trace_summary()
    assert wm_summary["triplets_stored"] > 0
    assert wm_summary["active_slots"] > 0

    ltm = EnhancedTripleHybridMemory(input_dim=32, output_dim=32, hg_slots=16, cgmn_slots=16, curved_slots=16)
    ltm(x, operation="write")
    for bank in ltm.qh_banks.values():
        summary = bank.trace_summary()
        assert summary["triplets_stored"] > 0
        assert summary["active_slots"] > 0


def test_cms_qh_bank_wires_and_persists_key_slot_map():
    store = ConsolidatedMemoryStore(
        ConsolidatedMemoryCfg(
            d_model=32,
            d_hyp=16,
            d_spher=16,
            d_fisher=8,
            d_phase=8,
        )
    )
    candidate = {
        "E": torch.randn(32),
        "H": torch.randn(16),
        "S": torch.randn(16),
        "P": (torch.rand(8), torch.randn(8)),
    }
    store.write("holo:key:1", candidate, alpha=0.35, src_info={"test": True})
    s = store.qh_slot_bank.trace_summary()
    assert s["triplets_stored"] > 0
    assert "holo:key:1" in store._qh_key_slot_map
    extra = store.extra_state_dict()
    assert "qh_key_slot_map" in extra
