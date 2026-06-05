# Context Compression, Parameter/Weight References, and Episodic Candidate Memory Extension

## Purpose

This extension adds a bounded context-memory layer to QDT-WM-MAAE.

It implements:

1. Context compression from `[B,T,D]` context tensors.
2. Optional response compression from generated response tensors.
3. Content signatures linking context and response.
4. Parameter and weight references related to the context/response path.
5. Episodic memory candidates for later consolidation into project/chat memory.
6. Optional explicit staging into `SharedSlotStore` and `QuantumHolographicStorage`.

## Source of truth

- Parent pack: `/mnt/data/qdt_wm_maae_wm_qd6a_quality_deepened_final_release_pack.zip`
- Parent SHA256: `1302773410ceb42172f9fc7b1b26411ff4ed3aad8f126c25d6c6f5de58f90dac`

## Safety doctrine

Default behavior is proposal-only. The extension does **not** mutate model weights and does **not** write to memory stores unless `allow_store=True` and `write_permission=True` are explicitly supplied.

Parameter references are metadata-only:

- parameter name
- module path
- shape
- dtype
- trainable flag
- bounded fingerprint
- relevance score

Full model weights are not copied into memory records.

## New source file

- `mnemonic_cortex/working_memory/context_compression_memory.py`

## Patched files

- `mnemonic_cortex/working_memory/wm_context_mount.py`
- `mnemonic_cortex/working_memory/__init__.py`

## New test file

- `tests/test_wm_context_compression_memory.py`

## Core APIs

```python
from mnemonic_cortex.working_memory import (
    ContextCompressionConfig,
    ContextCompressor,
    ContextParameterReferenceExtractor,
    ContextEpisodicMemoryBuilder,
    GeometryMountedContextBuffer,
)
```

### Proposal-only candidate creation

```python
builder = ContextEpisodicMemoryBuilder(ContextCompressionConfig(dim=hidden_dim))
result = builder.build_and_stage(
    context=context_tokens,
    response=response_tokens,
    model=model,
    project_id="project-id",
    chat_id="chat-id",
    episode_id="episode-id",
    allow_store=False,
)
```

### Explicit storage path

```python
result = builder.build_and_stage(
    context=context_tokens,
    response=response_tokens,
    shared_slot_store=wm.shared_slot_store,
    qh_storage=wm.qh_storage,
    allow_store=True,
    write_permission=True,
)
```

## Integration with context buffer

`GeometryMountedContextBuffer` now exposes:

- `compress_context(context, response=None)`
- `build_context_memory_candidate(...)`

These APIs preserve the existing `mount(...)` behavior.

## Full-depth adequacy gate

PASS conditions:

- bounded token windows
- `[B,T,D]` shape validation
- finite tensor validation
- response-link fingerprinting
- bounded parameter-reference extraction
- no weight mutation
- no store mutation by default
- explicit write permission required for staging
- shared-slot and QH-compatible staging path
- JSON-safe traces and candidates
