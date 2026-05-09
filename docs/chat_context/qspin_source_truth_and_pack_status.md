# QSPIN-BRIDGE Source Truth and Pack Status

## Active source of truth

- Archive: `/mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip`
- Repo source-truth folder: `source_of_truth/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack/`
- Status: QD6A is the active implementation source basis for QSPIN-BRIDGE.

## WM-7A status

- WM-7A content is included inside QD6A.
- WM-7A is superseded by QD6A wherever content differs.
- WM-7A is usable only for historical comparison, rollback reference, or delta audit.
- Do not use WM-7A as the implementation source of truth for QSPIN work.

## Repository observation

The `experimental` branch already contains WM-era working-memory source such as `mnemonic_cortex/working_memory/wm_quaternion_depth.py`. QD6A adds quality-deepening guard and quality subsystem files. If a QD6A source module is missing from the runtime tree, Codex should either add it from the source-truth folder or explicitly explain why it is intentionally deferred.

## Critical QD6A families to preserve

- Foundation/context modules.
- Depth and transformer modules.
- Attention modules.
- External memory/storage modules.
- Shared-slot/QH modules.
- Commit/cortex/compatibility modules.
- Guard modules.
- Quality subsystem.
- Release/docs/tests/benchmarks.
- Production/deferred hardening caveats.
