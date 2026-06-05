# PROD-1 Rollback Harness Design

The rollback harness is dry-run only. It validates evidence for feature flag disablement, kill-switch trip, previous baseline restoration, audit-log preservation, and no state mutation. Missing evidence blocks shadow activation.
