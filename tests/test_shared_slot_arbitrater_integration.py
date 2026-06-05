import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.memory.shared_slot_arbitrater import (
    ArbitrationDecision as LegacyArbitrationDecision,
)
from mnemonic_cortex.memory.shared_slot_arbitrater import (
    SharedSlotArbitrator as LegacySharedSlotArbitrator,
)
from mnemonic_cortex.memory.shared_slot_arbitrater import SlotReadRequest as LegacySlotReadRequest
from mnemonic_cortex.memory.shared_slot_arbitrator import ArbitrationDecision
from mnemonic_cortex.memory.shared_slot_arbitrator import SharedSlotArbitrator
from mnemonic_cortex.memory.shared_slot_arbitrator import SlotReadRequest


def test_legacy_arbitrater_module_re_exports_canonical_symbols():
    assert LegacySharedSlotArbitrator is SharedSlotArbitrator
    assert LegacyArbitrationDecision is ArbitrationDecision
    assert LegacySlotReadRequest is SlotReadRequest


def test_cortex_shared_memory_arbitrator_threshold_plumbing():
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_shared_memory_subsystem(
        num_slots=32,
        num_systems=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        overwrite_threshold=0.22,
        merge_threshold=0.71,
        quarantine_interference_threshold=0.93,
        contradiction_split_threshold=5,
    )

    subsystem = model.shared_memory_subsystem
    assert subsystem is not None
    assert subsystem.arbitrator.overwrite_threshold == 0.22
    assert subsystem.arbitrator.merge_threshold == 0.71
    assert subsystem.arbitrator.quarantine_interference_threshold == 0.93
    assert subsystem.update_engine.contradiction_split_threshold == 5
