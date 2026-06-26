import torch

from mnemonic_cortex.working_memory.wm_triple_hybrid_ltm_adapter import TripleHybridLTMExternalMemoryBank


def test_select_bank_indices_pins_fused_when_top_k_less_than_bank_count():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.1],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.1],
        ]
    )
    selected = TripleHybridLTMExternalMemoryBank._select_bank_indices(scores, top_k=4)
    keys = [TripleHybridLTMExternalMemoryBank._BANK_KEYS[i] for i in selected]

    assert len(selected) == 4
    assert "fused" in keys
    assert keys[-1] == "fused" or "fused" in keys


def test_gather_banks_preserves_selected_order():
    memory_state = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    slot_ids = [["a", "b", "c", "d", "e", "f"]]
    indices = [0, 2, 4, 5]

    state, sc, ids = TripleHybridLTMExternalMemoryBank._gather_banks(
        memory_state, scores, slot_ids, indices
    )

    assert state.shape == (1, 4, 4)
    assert sc.tolist() == [[6.0, 4.0, 2.0, 1.0]]
    assert ids == [["a", "c", "e", "f"]]
