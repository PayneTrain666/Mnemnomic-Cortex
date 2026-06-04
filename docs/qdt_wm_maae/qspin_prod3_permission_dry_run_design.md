# QH / Shared-Slot Permission Dry-Run Design

Permission checks are dry-run only. Shared-slot, QH, and external-memory read requests are simulated metadata checks. All writes are rejected. Interference-check and commit-gate-review metadata are required.
