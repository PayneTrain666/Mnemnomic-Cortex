# Limitations, Risks, and Future Work

## Where dragons lurk

1. **Non-differentiable memory writes & DDP hazards**  
   External memories are mutated under `torch.no_grad()`; in DDP this desynchronises replicas.
2. **Brute-force fractal distance**  
   Complexity is $O(B S M D)$ per scale; large `mem_slots` becomes the bottleneck.
3. **Hologram growth & interference**  
   Spectral energy can inflate; no explicit normalisation or collision management.
4. **Noisy light-bulb**  
   Single linear proj + z-score; trigger rate can drift on non-stationary data.
5. **Semantic–mechanic gap**  
   Ricci flow, ODE, phases lack tailored losses; training may under-use them.
6. **Temperature asymmetry**  
   CGMN uses scalar; HG supports per-batch; leads to uneven sharpness.
7. **Write gate calibration**  
   Importance threshold is arbitrary before training.
8. **Checkpoint versioning**  
   No schema; future refactors may break loading.
9. **CPU latency**  
   Heavy CUDA optimisations; CPU inference is slow.

## Clear, actionable improvements (prioritized)

### P0 – Correctness & multi-GPU safety

* **Synchronize external memory updates in DDP** – after each write call `torch.distributed.all_reduce()` on holograms, keys/values and `usage_counts` (or register them as buffers and broadcast).
* **Clamp & renorm holograms** – after `_holo_write` compute per-slot RMS; down-scale slots that exceed a cap and optionally whiten per-frequency bins.
* **Unit-test pack**
  * HG/CGMN lower effective temperature & increase top-k when `fire=True`.
  * Retrieval entropy decreases on fire events.
  * External writes are synchronised across ranks.
  * Shapes/devices/dtypes match across CPU/GPU.

### P1 – Performance & capacity

* **Approximate nearest-neighbour for fractal search** – coarse quantiser → shortlist, then refine or swap in FAISS IVF-PQ; reduces O(M) to O(M^α) with α ≪ 1.
* **Cache frequency-domain keys** – reuse FFT + entangle for repeated queries.
* **Micro-batch hologram updates** – accumulate for N steps and apply one fused `index_add_`.

### P2 – Learning signal & policy

* **Self-supervised recall loss** – contrastive / InfoNCE between cue and retrieved memory.
* **Learned write gate with straight-through estimator** – stochastic Bernoulli gate; gradients flow while writes remain side-effectful.
* **Calibrate light-bulb** – keep trigger rate in 5–15 % band with adaptive threshold; log histograms.
* **Per-sample temperature in CGMN** (optional) – mirror HG behaviour, fallback to scalar if unstable.

### P3 – Robustness & UX

* **Hologram collision control** – soft capacity per slot, penalise over-used slots; periodic k-means re-assignment to defrag.
* **Versioned checkpoints & state schema** – explicit versions for holograms, usage counts, thresholds, etc., validated on load.
* **Metric logging that matters** – trigger rate, effective temp, top-k usage, write LR/EMA, hologram RMS, retrieval entropy, collision rate, consolidation events.

---

#### Practical expectations

* **Capacity scaling** – linear until interference dominates; ANN search pushes threshold up.
* **Latency profile** – hot path is fractal `cdist` + FFTs; ANN shortlist + batched FFTs give biggest wins.
* **Training dynamics** – without recall objectives memory acts like residual; adding recall loss & learned gating activates the full stack.

