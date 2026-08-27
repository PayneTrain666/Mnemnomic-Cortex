# bAbI training

The bAbI runner uses the same stable training profile as the GPU copy task,
adapted for single-answer question answering.

## Default run

```powershell
python tools/babi_train_eval.py --config en-qa1
```

Important defaults:

- QDT working memory with the `single_gpu_8_12gb` profile;
- QSPIN live routing and all QSPIN write/commit permissions disabled;
- model width 160, four-layer task decoder, batch size 32;
- AdamW at `3e-4`, weight decay `0.02`;
- 10% warmup followed by cosine decay to 8% of the starting rate;
- AMP on CUDA, gradient normalization to `0.8`, clipping at `1.0`;
- 30 epochs, validation checkpoint selection, target accuracy `0.95`;
- JSONL metrics and resumable `best.pt` / `last.pt` checkpoints.

`--amp_dtype auto` prefers BF16 on CUDA devices that report native BF16
support and otherwise uses FP16. `--amp_dtype bf16` also falls back safely to
FP16 when BF16 is unavailable. Use `--report_capacity` to print literal
parameter, live-gradient, optimizer-state, publicly reported CPS rollback, and
CUDA peak-allocation bytes as separate values. Trainers also request fused SDPA
backends when CUDA is available; this is a runtime preference, not a weight
compression claim.

For copy/reverse metric meanings and the live training dashboard, see
`docs/copy_training_metrics_guide.md`.

## Hybrid capacity profile notes

Capacity features remain opt-in and measured separately:

- Trainable Parameter CPS exact consolidation, optional shared-template / low-rank
  / sparse-exception compression, and irreversible
  `finalize_trainable_parameter_consolidation()` / rollback-snapshot release.
- Parameter Loop shadow routing can reference CPS handles; its manifold
  scalar-equivalent estimates are never added to literal capacity.
- Task-decoder `shared_gqa` capacity profile and activation-checkpointing flags
  live on `CortexSeqModel` and default to the standard MultiheadAttention path.
- QDT-specific attention edits remain deferred until the QD6A source-truth
  inventory is available.
- Optimizer state resume fails closed when a CPS/layout fingerprint no longer
  matches.

The default dataset source is the maintained `Muennighoff/babi` mirror because
the original Facebook archive URL currently returns HTTP 404. The runner
filters the mirror by the task number in `--config`.

## Smoke run

```powershell
python tools/babi_train_eval.py `
  --epochs 1 `
  --max_train_examples 32 `
  --max_test_examples 16 `
  --d_model 32 `
  --qdt_hardware_profile compact `
  --qdt_num_slots 4 `
  --qdt_transformer_layers 1 `
  --disable_task_decoder `
  --no_amp
```

## Resume or evaluate

```powershell
python tools/babi_train_eval.py `
  --config en-qa1 `
  --resume_checkpoint logs/babi_checkpoints/last.pt

python tools/babi_train_eval.py `
  --config en-qa1 `
  --resume_checkpoint logs/babi_checkpoints/best.pt `
  --evaluate_only
```

The checkpoint records the vocabulary, model and optimizer state, scheduler,
AMP scaler, QDT/CPS architecture, epoch, and global optimizer step.

