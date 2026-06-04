# HGM v0.1 Integration Roadmap

## Immediate next step

```text
DEV-FLOW FINALIZE HGM-V0.1 — Preserve Hypergraph Manifold / Hyperset Probability Matrix Expansion Release State
```

## Roadmap items

1. **Finalize HGM v0.1 release state**
   - Preserve the final HGM v0.1 package, docs, tests, manifests, and archives.

2. **Audit QDT/WM bridge contracts**
   - Compare HGM bridge payloads and shared slot-lattice hooks with live QDT/WM package internals.
   - Keep the audit read-only.

3. **Design explicit write-capable stage**
   - Write execution must require a hard approval gate.
   - Transaction logs and rollback manifests must be mandatory.

4. **Train/evaluate learned embeddings**
   - Replace deterministic baseline embeddings only after evaluation harnesses are strong enough.

5. **Simulator-only robotics bridge**
   - Connect advisory robotics planning outputs to a simulator harness before any hardware integration.

6. **Production-readiness review**
   - Production remains blocked until write permissions, rollback verification, learned runtime validation, and integration tests are mature.

## Non-goals for v0.1

- No live QDT/WM writes.
- No slot-lattice mutation.
- No robotics actuator execution.
- No production enablement.
