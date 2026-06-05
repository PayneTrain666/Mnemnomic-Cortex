# PROD-5 Canary Test Plan

Canary categories:
- route_shape_canary
- payload_integrity_canary
- qh_no_write_canary
- shared_slot_no_write_canary
- external_memory_no_write_canary
- topology_no_execute_canary
- commit_gate_no_commit_canary
- audit_chain_canary
- redaction_canary
- concurrency_guard_canary
- timeout_bound_canary
- malformed_input_canary

Expected unsafe canaries pass only when the runtime blocks them fail-closed.
