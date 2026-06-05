# PROD-2 Observability / Audit / Metrics Design

Metrics, audit events, traces, dead-letter records, and diagnostic snapshots are metadata-only. Records contain reason codes and safe summaries only, never secrets or raw payloads. Duplicate audit IDs are idempotent.
