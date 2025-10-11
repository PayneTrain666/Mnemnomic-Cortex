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

## Actionable improvements

### P0 – Correctness & Multi-GPU safety  
* All-reduce external memories after writes.  
* Clamp and renorm holograms.

### P1 – Performance & Capacity  
* Approximate NN search (FAISS IVF-PQ).  
* Cache FFT keys & batched updates.

### P2 – Learning Signal & Policy  
* InfoNCE recall loss.  
* Learned write gate (STE).  
* Adaptive light-bulb threshold.

### P3 – Robustness & UX  
* Slot collision control.  
* Versioned checkpoints.  
* Rich metric logging.
