# QSPIN-PROD-5-QD6A Audit and Ship-Check

Ship condition:
- `python3 -S prod5_manual_runner.py` returns 0.
- Synthetic canaries pass.
- Safety regressions pass.
- Forbidden behavior is blocked fail-closed.
- Production activation remains disabled.

Decision:
SHIP QSPIN-PROD-5-QD6A as synthetic/sandbox-only live-adjacent bridge infrastructure if release test result reports fail_count = 0.
