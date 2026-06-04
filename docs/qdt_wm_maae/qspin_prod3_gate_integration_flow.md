# PROD-3 Gate Integration Flow

Integration order: production config, kill-switch, rollback dry-run, PROD-1 commit-gate dry-run, PROD-1 shadow activation, PROD-2 payload dry-run, PROD-2 shadow bus, PROD-2 guarded dispatch, PROD-3 commit approval simulation, payload roundtrip stub, permission dry-run, active-dry-run executor.
