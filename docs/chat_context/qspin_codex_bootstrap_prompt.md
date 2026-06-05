# QSPIN-BRIDGE Codex Bootstrap Prompt

Before working on QSPIN-BRIDGE, read these files in order:

1. `AGENTS.md`
2. `docs/chat_context/qspin_source_truth_and_pack_status.md`
3. `docs/chat_context/qspin_stage_0_to_8_manifest.md`
4. `docs/chat_context/qspin_dev_flow_v2_1_preferences.md`
5. `docs/chat_context/qspin_deferred_hardening_plan.md`

Then inspect the active code or source-of-truth files relevant to the task.

## Active source rule

- Active source of truth: QD6A release pack.
- WM-7A status: included inside QD6A, superseded where content differs, historical comparison / rollback / delta audit only.
- QSPIN-0 through QSPIN-8 are metadata/scaffold stages unless a later command explicitly authorizes guarded runtime activation.

## Safety defaults

- QSPIN disabled/inert by default.
- No live bridge routing.
- No payload transfer.
- No shared-slot writes.
- No external LTM/MANN/SPCP writes.
- No QH storage writes.
- No commit execution.
- No production activation.

## How to use this context in Codex

Tell Codex:

```text
Read AGENTS.md and docs/chat_context/qspin_codex_bootstrap_prompt.md first.
Use docs/chat_context as project context.
Use source_of_truth/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack/ as source-truth inventory if present.
Do not treat old chat context as executable code unless explicitly marked as source.
```
