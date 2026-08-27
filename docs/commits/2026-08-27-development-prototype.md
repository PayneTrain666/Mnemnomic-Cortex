# Commit record — development-prototype

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Branch | `development-prototype` |
| Type | `feat` (prototype drop; includes `fix` work) |
| Scope | `wm`, `ltm`, `training` |
| Status | Prototype / reviewable, not a production release |
| Parent | `f2e610b` Add trainable CPS and document active code paths |

## Labels

Industry-style labels for this change (GitHub + review filters):

| Label | Kind | Why it applies |
|---|---|---|
| `type:feature` | type | Inter-manifold attention, PSLS training, metrics dashboard |
| `type:fix` | type | Copy-task NaN/nonfinite recovery and apply-shadowing fix |
| `area:working-memory` | area | QDT-WM geometry-map attention and NaN sanitizers |
| `area:ltm` | area | Bank-read views, HG complex FP32, inter-memory stats |
| `area:training` | area | `copy_task_gpu_train.py`, metrics catalog/dashboard |
| `status:prototype` | status | Development branch; not production-ready |
| `safety:qspin-inert` | safety | No QSPIN live routing/writes/commits |
| `risk:medium` | risk | New residual mixers and checkpoint keys; gated identity at 0 |

## Summary

Stabilize copy/reverse training and add a residual inter-manifold attention path
that **monitors and uses** communications among QDT geometry-map depths, LTM
banks, MANN, SPCP, and optional PSLS views—without turning geometry maps into
native manifold charts and without activating QSPIN.

## Motivation

Copy-reverse training was dying on nonfinite logits (AMP/`ComplexHalf`, incomplete
snapshots) and CUDA eval faults. PSLS needed to amplify depth without breaking
decoder identity. Geometry maps were labels/biases only; inter-manifold attention
gives a safe way to watch and mix those communications before any native chart
projection work.

## Scope included

- QDT-WM + cortex inter-manifold attention (`ima_*` metrics); residual mix
  identity at 0.
- PSLS mount after CPS restore; gate mix `seq + gate * (norm(loop) - seq)`;
  consolidation after `optimizer.step()`.
- Trainable CPS / capacity reporting; benchmark capacity helpers.
- Copy-task trainer recovery, metrics JSONL, dashboard, docs.
- WM nonfinite sanitization (raise → `nan_to_num`).

## Scope excluded

- Generated run artifacts (`prod7_outputs/`, `logs/`).
- `pytorch_new` submodule dirty state.
- One-shot source-rewrite helpers `tools/_fix_wm_nan_inplace.py` and
  `tools/_sanitize_wm_nan_guards.py` (local maintenance scripts, not runtime).
- Native manifold projection from geometry maps (still labels + bias + this mixer).
- QSPIN live activation.

## Tests

- `tests/test_wm_inter_manifold_attention.py`
- `tests/test_wm_qd3a_module_contracts.py`
- `tests/test_wm2c_qdt_working_memory_assembly.py`
- `tests/test_parameter_storage_loop_stack.py`
- `tests/test_trainable_parameter_cps.py`
- `tests/test_training_metrics_dashboard.py`
- `tests/test_benchmark_model_capacity.py`

## Rollback

1. `git revert <this-commit>` on `development-prototype`, or reset the branch to
   `f2e610b` if the drop has not been shared further.
2. Resume copy-task from a pre-drop checkpoint **without** `ima_*` /
   `parameter_storage_loop_stack` keys; new modules initialize on missing keys.
3. Inter-manifold mix and PSLS gate are identity at 0 / unset stack, so disabling
   them is the fast operational rollback without a full revert.

## Safety

QSPIN stays inert. Inter-manifold attention does not write shared slots, LTM,
MANN, SPCP, or QH storage.
