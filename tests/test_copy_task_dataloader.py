import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.copy_task_dataloader import (
    CopyTaskDataConfig,
    CopyTaskMasteryModel,
    align_logits_targets,
    build_copy_task_dataloader,
    copy_task_collate_fn,
    sinusoidal_positional_encoding,
)
from benchmark.tasks import TOK2IDX


def test_copy_task_dataloader_shapes():
    cfg = CopyTaskDataConfig(
        n_samples=64,
        max_len=8,
        batch_size=8,
        val_ratio=0.25,
        enable_sinusoidal_encoder=True,
        enable_sinusoidal_decoder=True,
        encoder_d_model=32,
        encoder_heads=4,
        decoder_heads=4,
    )
    loader = build_copy_task_dataloader(cfg, split="train")
    batch = next(iter(loader))
    assert batch["src"].shape[0] == 8
    assert batch["tgt"].shape[0] == 8
    assert batch["src_pos"].shape == batch["src"].shape
    assert "meta" in batch


def test_sinusoidal_mastery_forward_and_loss():
    cfg = CopyTaskDataConfig(
        n_samples=16,
        max_len=6,
        encoder_d_model=32,
        encoder_heads=4,
        decoder_heads=4,
        encoder_layers=1,
        decoder_layers=1,
    )
    ds_loader = build_copy_task_dataloader(
        CopyTaskDataConfig(**{**cfg.to_dict(), "n_samples": 8, "batch_size": 4}),
        split="all",
    )
    batch = next(iter(ds_loader))
    model = CopyTaskMasteryModel(cfg)
    logits = model(batch["src"], batch["tgt"])
    loss = model.mastery_loss(logits, batch["tgt"], ignore_index=TOK2IDX["<pad>"])
    assert logits.shape[:2] == batch["tgt"].shape
    assert torch.isfinite(loss)


def test_curriculum_length_ramps():
    cfg = CopyTaskDataConfig(
        curriculum_enabled=True,
        curriculum_start_len=4,
        curriculum_end_len=12,
        curriculum_ramp_epochs=4,
        min_len=4,
        max_len=12,
    )
    assert cfg.curriculum_length(1) == 4
    assert cfg.curriculum_length(4) >= 8


def test_sinusoidal_pe_dim():
    pe = sinusoidal_positional_encoding(10, 32)
    assert pe.shape == (10, 32)


def test_align_logits_targets_sos():
    logits = torch.randn(2, 7, 40)
    tgt = torch.randint(0, 40, (2, 8))
    lo, ta = align_logits_targets(logits, tgt)
    assert lo.size(1) == ta.size(1)
