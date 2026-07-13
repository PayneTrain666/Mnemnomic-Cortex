# Trainable Shared-Parameter CPS

## Purpose

The Trainable Parameter CPS is a second consolidated parameter store. It is
separate from:

- the existing token/concept CPS (`ConsolidatedParamStore`), and
- the parameter-loop stack, which stores trainable summary slots and
  scalar-equivalent capacity estimates but does not own model weights.

This store owns real weights used by `Linear`, `Embedding`, and supported
attention modules. Gradients flow from model output through the adapter into the
canonical CPS-owned tensor.

## What “density” means here

An exact arbitrary weight containing N independent scalar values requires N
physical scalar values. Manifold addressing cannot create additional independent
trainable numbers without an actual representation for them.

The store therefore reports two different ideas honestly:

1. **Literal physical capacity** — actual scalar count and bytes.
2. **Real compression** — fewer physical scalars through:
   - preserving existing tied parameters as one canonical tensor;
   - recognizing truly identical tensors;
   - a shared trainable template plus trainable low-rank residual factors.

The product-manifold “effective storage” estimate remains useful for addressing
and representation analysis, but is not reported as literal trainable capacity.

## Consolidation lifecycle

1. **Discover** eligible trainable modules and explain every rejection.
2. **Group** parameters by role, shape, device, and dtype.
3. **Stage exact** canonical flat storage without changing the live model.
4. **Evaluate** numerical output equivalence and gradient/device contracts.
5. **Commit exact** adapters only if validation passes.
6. **Propose compression** for compatible same-shape matrix cohorts.
7. **Evaluate compression** against reconstruction and model-output tolerances.
8. **Commit or reject**; a failed compressed proposal leaves exact storage active.
9. **Rollback** restores the original modules and parameters.

## Optimizer and checkpoint rules

- Structural consolidation should happen before optimizer creation.
- Optimizer parameters are deduplicated by object identity.
- A runtime commit must explicitly migrate supported optimizer state or reject
  the operation and require optimizer reconstruction.
- Checkpoints store the canonical tensors plus an architecture/binding manifest.
  The binding architecture is rebuilt before tensor state is loaded.

## Safety boundaries

- Disabled by default.
- Does not activate QSPIN.
- Does not write shared memory slots, LTM/MANN/SPCP, or QH storage.
- Does not use the WM system commit gate to mutate memory.
- Existing token CPS and parameter-loop behavior remain unchanged.
- Compression is optional; exact storage is always the initial committed form.

## Capacity report

The runtime report includes:

- eligible original scalar count;
- canonical exact scalar count;
- unique scalar count after true sharing;
- compressed template and factor scalar counts;
- real compression ratio;
- bytes by dtype;
- rough gradient and Adam/AdamW training-state cost;
- per-cohort reconstruction and output errors;
- rejected module reasons and handle ownership.

