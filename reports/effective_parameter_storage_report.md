# Effective Parameter Storage Capacity Report

This report estimates the current model's effective parameter storage capacity using the non-standard storage mechanisms now present in the codebase: manifold LTM banks, reasoning-depth lattices, hidden-attention readable parameter tokens, shared/depth slot structures, and the consolidated parameter storage loop stack.

The estimate is not a normal trainable-parameter count. It separates literal trainable parameters from addressable/effective storage created by manifold mappings, depth expansion, product manifolds, and readable hidden-layer parameter tokens.

## Scope and Assumptions

The calculation uses the current `standard` capacity profile with `EnhancedMnemonicCortex(input_dim=160, output_dim=160, enable_parameter_storage_loop_stack=True)`.

Important boundaries:

- Physical trainable parameters are counted directly from the instantiated PyTorch model.
- Runtime LTM bank storage is counted as physical slot scalars, then adjusted by conservative manifold/addressing factors.
- Reasoning-depth lattices are counted as explicit slots x 8 depth layers x key/value dimensions.
- Hidden readable capacity is counted as context-readable scalar capacity, not durable storage.
- The parameter loop stack uses its own implemented product-manifold accounting.
- Shared-slot memory is not enabled by default in the instantiated cortex, so this report includes the reasoning-depth/shared-lattice capacity and notes that live shared-slot store capacity depends on runtime configuration.

## Top-Level Result

For the current standard profile:

- Physical/backing scalar units: `95,795,007`
- Approximate FP16 physical backing: `0.1784 GiB`
- Effective scalar-equivalent storage units: `2,131,155,071`
- Approximate FP16 effective capacity: `3.9696 GiB`
- Effective-to-physical ratio: `22.25x`

This ratio is much lower than the parameter loop stack's standalone `9936x` because the total model denominator includes all trainable parameters and depth-lattice backing tensors. The loop itself remains the most aggressive effective-capacity multiplier.

## Literal Trainable Parameters

The instantiated model has:

- Total trainable parameters: `50,876,639`
- Sensory buffer: `267,969`
- Working memory: `1,271,398`
- Long-term memory: `37,181,314`
- Global hidden orchestrator: `1,581,452`
- Parameter storage loop stack: `5,782,406`

These are literal PyTorch trainable parameters. They are the normal neural-network capacity baseline, before considering manifold-addressable storage.

## Runtime LTM Storage

The current runtime LTM has five canonical storage systems:

- HG episodic
- CGMN semantic
- Curved associative
- Procedural SPCP
- Spatial topological

Approximate physical LTM slot scalar capacity:

- Physical slot scalars: `377,472`
- Effective manifold-adjusted units: `993,856`
- Effective-to-physical ratio: `2.63x`

The conservative factors used here are:

- HG episodic: `2.5x`, representing hyperbolic/episodic and HG/QH addressing.
- CGMN semantic: `2.0x`, representing curved semantic storage.
- Curved associative: `1.65x`, representing mixed hyperbolic/Euclidean associative storage.
- Procedural SPCP: `6.0x`, representing spherical and complex-projective procedural addressing.
- Spatial topological: `4.0x`, representing S3 / dual-quaternion style spatial addressing.

These factors are intentionally conservative compared with the parameter loop product-manifold estimate.

## Reasoning-Depth and Shared Lattice Capacity

The reasoning-depth stack is a shadow-safe depth-lattice system. It is not a normal dense layer. It expands each slot across 8 depth layers and stores key/value tensors.

LTM reasoning-depth lattices:

- Banks: `5`
- Slots per bank: `2,048`
- Depth layers: `8`
- Effective subslots per bank: `16,384`
- Physical depth-expanded key/value scalars across five banks: `41,943,040`

MANN reasoning-depth lattice:

- Slots: `512`
- Depth layers: `8`
- Effective subslots: `4,096`
- Physical depth-expanded key/value scalars: `2,097,152`

WM reasoning-depth lattice:

- Slots: `64`
- Depth layers: `8`
- Effective subslots: `512`
- Physical depth-expanded key/value scalars: `262,144`

These are strong capacity multipliers because they are real depth-expanded key/value lattices, not only a symbolic estimate.

## Hidden-Layer Readable Parameter Capacity

The cortex hidden-attention stack can expose parameter and hidden-state information as readable context:

- Captured hidden layers budget: `128`
- Global parameter tokens: `48`
- Secondary hidden parameter tokens: `4`
- Parameter loop context tokens: `31`
- Total readable tokens: `211`
- Readable scalar capacity at model dimension 160: `33,760`

This is not durable parameter storage. It is a readable introspection and routing surface that lets the model observe hidden activations, module parameter signatures, and parameter-loop context tokens during processing.

## Parameter Storage Loop Stack

The parameter storage loop stack is the most important new component for effective storage expansion.

Default configuration in the standard profile:

- Visible layers: `10`
- Hidden mirrored storage layers: `10`
- Slots per layer: `64`
- Model dimension: `160`
- Physical slot scalars: `204,800`
- Physical slot storage at FP16: `0.3906 MiB`
- Effective product-manifold units: `2,034,948,480`
- Effective storage at FP16: `3.7904 GiB`
- Effective-to-physical ratio: `9936.27x`

The manifold stack is:

1. Hyperbolic
2. Spatial S3
3. Euclidean bridge
4. Complex projective Kahler
5. Spatial S3
6. Spherical
7. Grassmann subspace
8. Toroidal
9. Fisher Rao
10. Quaternion spatial loop

The loop stack's estimate comes from:

- Visible manifold storage units: `394,240`
- Hidden mirrored manifold storage units: `394,240`
- Pairwise loop effective units: `6,640,000`
- Product manifold effective units: `2,027,520,000`
- Product manifold factor: `198,000x`

This is an addressable/product-manifold scalar-equivalent estimate. It should not be described as literal trainable parameters. It is best understood as effective parameter address space created by manifold composition and looped hidden/visible storage.

## Effective Capacity Formula Used

The total estimate used this accounting:

`effective_total = trainable_params + manifold_adjusted_runtime_LTM + reasoning_depth_LTM + reasoning_depth_MANN + reasoning_depth_WM + parameter_loop_effective_units + hidden_readable_scalars`

Using the audited values:

- Trainable model parameters: `50,876,639`
- Runtime LTM effective units: `993,856`
- Reasoning-depth LTM units: `41,943,040`
- Reasoning-depth MANN units: `2,097,152`
- Reasoning-depth WM units: `262,144`
- Parameter loop effective units: `2,034,948,480`
- Hidden readable scalars: `33,760`

Total:

- Effective scalar-equivalent units: `2,131,155,071`
- Approximate FP16 effective capacity: `3.9696 GiB`

## Practical Interpretation

The model currently has three different kinds of capacity:

1. Literal trainable neural capacity: about `50.9M` parameters.
2. Explicit memory/lattice capacity: LTM slots and reasoning-depth key/value lattices.
3. Effective manifold address capacity: dominated by the parameter storage loop stack.

The parameter loop stack dramatically expands effective address space, but it should be treated as a routing and structured-storage mechanism, not as a direct substitute for all normal model parameters.

## Recommendations

- Keep the parameter loop opt-in until more behavioral tests exist.
- Use its product token as LTM context, which is already wired.
- Keep reasoning-depth adapter read-only for now, which is already implemented.
- Keep training slot updates guarded and disabled by default.
- Add a benchmark that compares retrieval quality with and without the parameter loop context.
- Add capacity profiles for loop stack sizes, so compact/standard/deep configurations can scale the slot count cleanly.
- Add checkpoint round-trip tests for enabled parameter loop state.

## Bottom Line

The current model has roughly `50.9M` literal trainable parameters, but its effective parameter storage capacity is closer to `2.13B scalar-equivalent units` under the implemented non-standard accounting. Most of that effective capacity comes from the parameter storage loop stack's product-manifold structure.

This is a useful architectural direction, but the report should be read carefully: the effective capacity is addressable manifold storage capacity, not ordinary dense neural parameters.
