# PROD-2 Active Patching Report

P001: Shadow bus could be mistaken for live adapter dispatch. Patch: dispatch result rejects live routing, payload transfer, writes, commits, and production activation.

P002: Payload dry-run could leak payloads. Patch: raw payload and raw tensor fields are rejected, and traces use metadata-only safe hashes.

P003: Guarded dispatch could be mistaken for topology execution. Patch: simulator returns simulation metadata only and blocks missing approvals.

P004: Observability could leak data. Patch: traces, audit, and dead-letter records reject raw payload and secret flags.
