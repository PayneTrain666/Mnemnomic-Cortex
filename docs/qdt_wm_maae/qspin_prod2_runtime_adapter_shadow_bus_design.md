# Runtime Adapter Shadow Bus Design

The shadow bus registers metadata-only adapter endpoints and simulates dispatch only after PROD-1 gates pass. It is idempotent, fail-closed, non-mutating, and exposes safe trace/audit summaries only.
