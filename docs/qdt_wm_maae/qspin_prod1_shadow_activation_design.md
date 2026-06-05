# PROD-1 Shadow Activation Design

Shadow activation is feature-flagged, idempotent, fail-closed, trace-safe, and non-mutating. A request is allowed only when source consideration is complete, rollback evidence passes, kill-switch allows shadow operation, commit-gate dry-run allows shadow inspection, and no unsafe feature flags are requested. No live data routing or payload transfer occurs.
