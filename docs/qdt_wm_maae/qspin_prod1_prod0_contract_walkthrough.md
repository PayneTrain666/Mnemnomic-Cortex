# PROD-0 Contract Walkthrough

PROD-0 introduced qspin_production_config.py and qspin_production_plan.py as production-track planning contracts. PROD-1 preserves disabled runtime activation, kill-switch requirement, dry-run-before-active doctrine, commit-gate approval requirement, rollback requirement, observability requirement, source consideration coverage, and production caveats. PROD-1 extends PROD-0 by adding shadow-only callable controllers and dry-run validators; it does not activate production routing.
