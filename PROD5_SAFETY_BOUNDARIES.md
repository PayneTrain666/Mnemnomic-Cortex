# PROD-5 Safety Boundaries

The PROD-5 implementation is live-adjacent only in shape. It is not live-adjacent in authority.

Explicitly blocked:
- live routing
- real payload transfer
- topology execution
- real shared-slot writes
- QH writes
- external-memory writes
- commit execution
- production activation
- production config mutation
- networking

All blocked attempts return FAIL_CLOSED with reason code `SAFETY_BOUNDARY_BLOCK` unless they are idempotent replay skips.
