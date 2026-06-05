# HGM-0B — Hyperset Probability Expander

HGM-0B adds the first runtime layer for **Hyperset Probability Matrix Expansion**.
It is additive, dependency-light, and remains isolated under:

```text
mnemonic_cortex/hypergraph_manifold/
```

## Runtime goal

The HGM-0B runtime converts either typed `MutationToken` records or matrix-like
payloads into validated probability payloads, applies explicit normalization,
and extracts top-k scenario candidates with trace records.

## Probability tensor ranks

The runtime recognizes these contracts:

| Rank | Contract | Meaning |
|---:|---|---|
| 2 | `P[v,m]` | variable by mutation/magnitude matrix |
| 3 | `P[v,m,d]` | adds depth layer |
| 4 | `P[v,m,d,c]` | adds context cluster |
| 5 | `P[v,m,d,c,t]` | adds time |
| 6 | `P[v,m,d,c,t,a]` | adds action candidate |

The mutation axis is always `m` and is used for row/mutation-axis normalization.

## Normalization modes

- `NONE`: validate and preserve values.
- `ROW_STOCHASTIC`: normalize each mutation-axis row/group to sum to `1.0`.
- `MUTATION_AXIS` and `ROW`: compatibility aliases for row-stochastic behavior.
- `GLOBAL_SUM`: normalize the full payload to global sum `1.0`.
- `GLOBAL`: compatibility alias for `GLOBAL_SUM`.
- `SOFTMAX`: stable softmax over the mutation axis. This mode can accept negative logits.

## Top-k scenario extraction

`extract_top_k_scenarios()` flattens the probability tensor into probability cells,
sorts by descending score/probability, and breaks ties deterministically by source
index tuple. Each returned `ScenarioCandidate` includes:

- candidate ID
- variable ID
- magnitude bin ID
- optional depth/context/time/action indices
- probability and score
- source index tuple
- trace ID

`k <= 0` returns an empty result with a warning trace rather than raising.

## Connection to HGM-1

HGM-0B extracts probability cells as candidate scenario atoms. HGM-1 should bind
these atoms into multi-node hyperedges, score coherence, detect conflict bundles,
and detect opportunity bundles. HGM-0B is therefore the probability-cell substrate
for future hyperedge scenario binding.
