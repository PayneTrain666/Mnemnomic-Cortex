# REASON-2A Consolidation Gate Design

The `ShadowConsolidationGate` evaluates LTM consolidation proposals without committing them.

Decision states:

- `shadow_only`
- `denied`
- `quarantined`
- `commit_ready`

`commit_ready` is not a write. It only marks an item as eligible for a later explicit commit path.

Default behavior is shadow-only. Permanent memory mutation is not implemented in REASON-2A.
