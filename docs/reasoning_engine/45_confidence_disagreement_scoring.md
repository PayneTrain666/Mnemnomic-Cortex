# REASON-2B Confidence / Disagreement Scoring

The scorer accepts finite score tensors and returns:

- `confidence`
- `disagreement`
- `support_mass`
- entropy and normalized entropy
- PAAMA-X-compatible confidence/disagreement metadata

The algorithm is intentionally bounded and deterministic. It uses a stable softmax over support scores and combines margin, entropy, and support mass. It never mutates input tensors or memory stores.
