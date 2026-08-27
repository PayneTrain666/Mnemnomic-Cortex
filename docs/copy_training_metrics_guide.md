# Copy / reverse training metrics guide

This guide explains the metrics printed by `tools/copy_task_gpu_train.py` and
stored in the run's `metrics_jsonl` file. Use it with the live dashboard:

```powershell
python tools/training_metrics_dashboard.py --metrics_jsonl logs/copy_reverse_smoke/metrics.jsonl
```

Or start training with the dashboard attached:

```powershell
python tools/copy_task_gpu_train.py --copy_task_mode reverse --open_metrics_dashboard ...
```

## Where metrics come from

| Source | What you see |
|--------|--------------|
| Compact terminal line `[train ep=...]` | Core loss/accuracy/grad/lr snapshot |
| Full dump `[train_full_metrics] {...}` | Complete JSON event for that step |
| `metrics_jsonl` | Durable stream of `run_start`, `train_step`, `epoch_end` |
| Live dashboard | Charts + advice polling the JSONL every 2 seconds |

## Core terminal line

Example:

```text
[train ep=01 gstep=0010/0120 len=08] loss=3.5795 mean=3.7512 acc=0.1793 seq_acc=0.0000 recall=2.0960 cms=0.0014 pre_grad=539.3589 grad=539.3589 lr=0.000275 topo_fit=0.2090 clip_rate=1.000
```

| Field | Meaning | How to read it |
|-------|---------|----------------|
| `ep` | Current epoch | Curriculum stage counter |
| `gstep` | Global optimizer step / budget | Progress through `--total_steps` |
| `len` | Current sequence length | Accuracy often drops when this jumps (8→16) |
| `loss` | Current-batch cross-entropy | Immediate noise; look at trend |
| `mean` | Running mean loss in the epoch | Better trend signal than single-step loss |
| `acc` | Token accuracy | Rises before exact sequence accuracy |
| `seq_acc` | Exact full-sequence accuracy | Strict success metric; late to rise |
| `recall` | Memory recall auxiliary loss | Should usually fall as retrieval stabilizes |
| `cms` | CMS/CPS auxiliary loss | Small positive is normal |
| `pre_grad` | Gradient norm before clip/normalize | Explosions show here first |
| `grad` | Gradient norm after control | Compare to `--target_grad_norm` |
| `lr` | Current AdamW learning rate | Warmup up, cosine down |
| `topo_fit` | Topology fitness EMA | Higher generally healthier dynamics |
| `clip_rate` | Fraction of recent steps that clipped | Near `1.0` means updates are always oversized |

## Validation / epoch summary

| Field | Meaning |
|-------|---------|
| `train_mean_loss`, `train_acc`, `train_seq_acc` | Full-epoch training aggregates |
| `val_loss_curriculum_len`, `val_acc_curriculum_len` | Held-out metrics at the active length (default best-checkpoint metric) |
| `val_*_len8`, `val_*_len16` | Fixed-length probes so curriculum jumps do not hide forgetting |
| `val_seq_acc_*` | Exact-sequence validation rates |

## Memory-bank metrics (`hg_`, `cgmn_`, `curved_`)

These describe long-term memory health, not token labels.

| Suffix | Meaning |
|--------|---------|
| `active_slots` | Slots currently active |
| `usage_mean` / `usage_max` | How much slots are being reused; huge max vs mean can mean collapse |
| `temp` | Retrieval temperature (lower = sharper) |
| `topk_base` | Top-k slots retrieved |
| `geom_w_*` | Soft weights over geometry heads (euclidean/hyperbolic/spherical/torus/fractal/cp) |
| `topology_fitness_ema` | Bank-local fitness |
| `holo_rms_*` | Quantum-holographic code energy stats |
| `importance_mean` | Mean slot importance |
| `lb_top1_avg` | Lightbulb / top-1 intensity |

## LTM router metrics

| Field | Meaning |
|-------|---------|
| `ltm_router_hg/cgmn/curved/spcp` | Soft choice among memory banks |
| `ltm_router_lightbulb_intensity` | Fire/lightbulb feature into the router |
| `ltm_router_cons_novelty` | Novelty/consolidation feature |
| `ltm_inter_*` | Inter-bank attention / gate diagnostics |

If one router weight stays above ~0.75 for many steps, other banks may be underused.

## Diagnostics EMA metrics (`diag_ema_*`)

Smoothed bridge, write-gate, fire-rate, recall, sensory-norm, and CPS-agreement
signals. Use them for regime changes, not as absolute pass/fail scores.

## Practical advice cheatsheet

- Loss down + token acc up, seq_acc still ~0: normal early learning; keep going.
- Val acc drops exactly when `len` increases: curriculum shock; compare `val_acc_len8` vs `val_acc_len16`.
- `clip_rate≈1` and huge `pre_grad`: lower `--lr` or raise `--target_grad_norm` carefully.
- BF16 AMP can break complex QH paths (`view_as_complex`); prefer `--amp_dtype fp16` for this stack.
- Recall falling while token acc is still modest: memory path is stabilizing before the decoder fully masters reverse.

The authoritative machine-readable glossary lives in
`tools/training_metrics_catalog.py` and is served by the live dashboard.
