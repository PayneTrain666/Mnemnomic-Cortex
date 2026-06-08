import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.models import CortexSeqModel
from benchmark.tasks import TOK2IDX, VOCAB_SIZE
from tools.copy_task_gpu_train import (
    _shared_slot_episodic_available,
    _shared_slot_training_available,
    _write_consolidation_snapshot_to_shared_slots,
    _write_successful_copy_episodes,
)


def _build_full_stack_model(d_model: int = 32) -> CortexSeqModel:
    model = CortexSeqModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        task_decoder_enabled=True,
        cms_enabled=True,
    )
    model.cortex.enable_cps_cms_full_stack(
        vocab_size=VOCAB_SIZE,
        cms_senses=3,
        enable_broker=True,
        enable_advanced=True,
        enable_reasoning_bridge=True,
        enable_qdt_wm_bridge=True,
    )
    return model


def test_shared_slot_helpers_write_episodes_and_consolidation():
    torch.manual_seed(0)
    model = _build_full_stack_model(d_model=32)
    assert _shared_slot_training_available(model.cortex)
    assert _shared_slot_episodic_available(model.cortex)

    pad = TOK2IDX["<pad>"]
    src = torch.tensor(
        [
            [3, 4, 5, pad],
            [6, 7, pad, pad],
        ],
        dtype=torch.long,
    )
    row_ok = torch.tensor([True, False])
    valid_rows = torch.tensor([True, True])

    n_ep = _write_successful_copy_episodes(
        model,
        src=src,
        row_ok=row_ok,
        valid_rows=valid_rows,
        global_step=12,
        epoch=1,
        cur_len=8,
        max_episodes=4,
    )
    assert n_ep == 1
    metrics = model.cortex.get_metrics()
    assert metrics["shared_mem_used_slots"] > 0.0
    assert metrics["hg_episodic_records"] >= 1.0

    n_cons = _write_consolidation_snapshot_to_shared_slots(
        model,
        src=src,
        global_step=12,
        epoch=1,
    )
    assert n_cons >= 1
    metrics_after = model.cortex.get_metrics()
    assert metrics_after["shared_mem_used_slots"] >= metrics["shared_mem_used_slots"]
