# PROD-5 Shadow Runtime Specification

The shadow harness accepts production-shaped envelopes but executes only local synthetic planning. It emits audit events and synthetic route IDs. It never transfers a payload or writes to memory.

Modes:
- SHADOW_IDLE
- SHADOW_VALIDATE
- SHADOW_PLAN
- SHADOW_EXECUTE_SYNTHETIC
- SHADOW_AUDIT
- SHADOW_FAIL_CLOSED

Validation is fail-closed and includes config posture, payload size, route/target markers, forbidden flags, and supported mode checks.
