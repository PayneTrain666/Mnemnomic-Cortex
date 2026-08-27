from __future__ import annotations

import json
from pathlib import Path

from tools.training_metrics_catalog import describe_metric, interpret_training
from tools.training_metrics_dashboard import build_snapshot


def test_core_metrics_have_glossary_entries():
    for key in ("loss", "acc", "seq_acc", "pre_grad_norm", "val_acc_curriculum_len"):
        meta = describe_metric(key)
        assert meta["meaning"]
        assert meta["group"]


def test_interpret_training_produces_advice_and_trends():
    events = [
        {"kind": "run_start", "schema_version": 1},
        {
            "kind": "train_step",
            "global_step": 10,
            "loss": 3.5,
            "mean_loss": 3.7,
            "acc": 0.10,
            "seq_acc": 0.0,
            "recall_loss": 2.0,
            "pre_grad_norm": 400.0,
            "clip_hit_rate": 1.0,
            "lr": 3e-4,
            "topology_fitness_ema": 0.2,
            "seq_len": 8,
        },
        {
            "kind": "train_step",
            "global_step": 20,
            "loss": 3.2,
            "mean_loss": 3.4,
            "acc": 0.15,
            "seq_acc": 0.0,
            "recall_loss": 1.0,
            "pre_grad_norm": 350.0,
            "clip_hit_rate": 1.0,
            "lr": 2.5e-4,
            "topology_fitness_ema": 0.22,
            "seq_len": 8,
        },
        {
            "kind": "train_step",
            "global_step": 30,
            "loss": 2.9,
            "mean_loss": 3.1,
            "acc": 0.20,
            "seq_acc": 0.01,
            "recall_loss": 0.4,
            "pre_grad_norm": 300.0,
            "clip_hit_rate": 0.9,
            "lr": 2.0e-4,
            "topology_fitness_ema": 0.25,
            "seq_len": 8,
        },
    ]
    result = interpret_training(events, target_grad_norm=0.8)
    assert result["highlights"]["loss_trend"] == "down"
    assert result["highlights"]["acc_trend"] == "up"
    assert any(item["title"] for item in result["advice"])


def test_build_snapshot_reads_smoke_metrics_if_present():
    path = Path("logs/copy_reverse_smoke/metrics.jsonl")
    if not path.exists():
        return
    snapshot = build_snapshot(path)
    assert snapshot["event_count"] > 0
    assert "advice" in snapshot
    assert snapshot["glossary"]
    json.dumps(snapshot["highlights"])
