# PROD-5 Observability Specification

Observability is local, structured, and redaction-safe. It includes:
- MetricEvent
- AuditEvent
- SpanEvent
- counters for fail-closed, skips, canaries, remediations, audit events, metrics, and spans
- lineage tags
- stage tags
- source tags

Secret-shaped values are redacted before serialization.
