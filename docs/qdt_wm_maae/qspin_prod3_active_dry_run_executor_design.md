# Active-Dry-Run Executor Design

The executor simulates bridge execution order after all gates pass. It is feature-flagged, idempotent, non-mutating, and fails closed. It does not call live QD6A runtime modules or perform real routing, transfer, writes, commits, or production activation.
