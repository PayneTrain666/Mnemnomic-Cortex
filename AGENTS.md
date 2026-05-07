# Codex Instructions — Mnemnomic-Cortex / QSPIN-BRIDGE

## Load order before modifying code

1. Read `docs/chat_context/qspin_codex_bootstrap_prompt.md`.
2. Read `docs/chat_context/qspin_source_truth_and_pack_status.md`.
3. Read `docs/chat_context/qspin_stage_0_to_8_manifest.md`.
4. Read `docs/chat_context/qspin_dev_flow_v2_1_preferences.md`.
5. If touching QSPIN or QD6A WM code, read the relevant source-of-truth inventory under `source_of_truth/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack/`.

## Standing rules

- QD6A is the active implementation source of truth for QSPIN-BRIDGE work.
- WM-7A is included inside QD6A but is historical-only where QD6A differs.
- Keep QSPIN disabled/inert unless a future stage explicitly authorizes guarded activation.
- Do not bypass QD6A context, triplet, attention, external-memory, shared-slot, QH, guard, compatibility, cortex, quality, or commit-gate semantics.
- Preserve source-consideration matrices, audit-pack, ship-check, and full-content-printout requirements for DEV-FLOW work.
- Do not treat chat context as executable code unless a file is explicitly marked as source code.
- Prefer small, reviewable commits with tests and rollback notes.
