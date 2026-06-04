# Trace-Safe Payload Dry-Run Design

Payload dry-run accepts envelope metadata only: kind, source, target, declared shape, dtype, norm band, and budget. Raw payload and tensor fields are rejected. Safe hashes are derived from metadata only.
