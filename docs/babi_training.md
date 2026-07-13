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

