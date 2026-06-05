# PROD-2 Gate Integration Flow

Required flow: PROD-1 config -> kill-switch enabled -> rollback dry-run evidence -> commit-gate dry-run approval -> shadow activation approval -> payload dry-run approval -> shadow bus simulation -> guarded dispatch simulation. Failure at any step blocks downstream simulation.
