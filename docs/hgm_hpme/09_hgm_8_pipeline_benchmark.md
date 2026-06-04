# HGM-8 — Runtime Embedding Trainer, Safe Replay, and Pipeline Benchmark

HGM-8 adds the first end-to-end evaluation layer for the HGM/HPME stack. It is
still evaluation-first: it does not perform live QDT/WM writes, does not mutate
working-memory internals, and does not require Torch or NumPy.

## Components

- `runtime_embedding_trainer.py` builds deterministic runtime embedding records
  from HGM-5 baseline embeddings and HGM-7 transaction/recovery/execution records.
- `safe_write_replay.py` evaluates HGM-7 transaction logs as replay evidence.
  Replay means checking log safety and consistency; it does not write memory.
- `pipeline_benchmark.py` combines embedding coverage, replay safety, trace
  observability, deterministic behavior, and no-live-write guarantees into a
  small benchmark score.
- `hgm8_result.py` stores typed outputs for trainer, replay, benchmark, and
  high-level HGM-8 evaluation.

## Safety posture

HGM-8 is read-only and replay-only. It consumes prior stage records and produces
metrics. It never calls QDT/WM write paths, never emits actuator commands, and
never integrates with robotics hardware.

## Future connection

HGM-8 prepares HGM-9 by providing benchmarkable end-to-end artifacts for runtime
integration evaluation, slot-lattice replay benchmarks, and production readiness
gates.
