# PROD-5 Ship Check

Required ship condition:
- manual synthetic runner passes with fail_count = 0
- forbidden actions are blocked fail-closed
- missing sources are structured skips only
- no live behavior is enabled

Ship posture after generation: SHIP as synthetic/sandbox-only live-adjacent bridge infrastructure if tests pass.
