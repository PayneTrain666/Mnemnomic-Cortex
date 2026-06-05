import json

import torch
import torch.nn as nn

from mnemonic_cortex.working_memory import (
    ContextCompressionConfig,
    ContextCompressor,
    ContextEpisodicMemoryBuilder,
    ContextParameterReferenceExtractor,
    GeometryMountedContextBuffer,
    SharedSlotStore,
    SharedSlotStoreConfig,
    QuantumHolographicStorage,
    QuantumHolographicStorageConfig,
    wm_context_compression_contract,
)


class TinyContextModel(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.context_proj = nn.Linear(dim, dim)
        self.response_attention = nn.Linear(dim, dim)
        self.unrelated = nn.Linear(dim, dim)

    def forward(self, x):
        return self.response_attention(self.context_proj(x))


def test_context_compressor_shapes_and_serialization():
    cfg = ContextCompressionConfig(dim=16, top_k_context_tokens=3, top_k_response_tokens=2)
    compressor = ContextCompressor(cfg)
    context = torch.randn(2, 5, 16)
    response = torch.randn(2, 4, 16)

    out = compressor(context, response)

    assert out.compressed_context.shape == (2, 16)
    assert out.compressed_response.shape == (2, 16)
    assert out.episode_vector.shape == (2, 16)
    assert torch.isfinite(out.episode_vector).all()
    assert out.context_top_indices.shape == (2, 3)
    assert out.response_top_indices.shape == (2, 2)
    json.dumps(out.to_dict())


def test_parameter_reference_extractor_is_bounded_and_weight_aware():
    cfg = ContextCompressionConfig(dim=16, max_parameter_refs=3)
    extractor = ContextParameterReferenceExtractor(cfg)
    model = TinyContextModel(dim=16)

    refs = extractor.extract(model, parameter_hints=["context", "attention"])

    assert 1 <= len(refs) <= 3
    assert any("weight" in r.reference_kind for r in refs)
    for ref in refs:
        payload = ref.to_dict()
        assert payload["paamax_metadata"]["full_weight_tensor_stored"] is False
        assert payload["paamax_metadata"]["parameter_mutation"] is False
        json.dumps(payload)


def test_context_memory_builder_creates_proposal_without_store_mutation():
    cfg = ContextCompressionConfig(dim=16, max_parameter_refs=8)
    builder = ContextEpisodicMemoryBuilder(cfg)
    model = TinyContextModel(dim=16)
    context = torch.randn(2, 6, 16)
    response = torch.randn(2, 5, 16)

    result = builder.build_and_stage(
        context=context,
        response=response,
        model=model,
        project_id="qspin-bridge",
        chat_id="chat-001",
        episode_id="episode-001",
        context_map="episodic",
        task_hints=["context", "response"],
        allow_store=False,
    )

    payload = result.to_dict()
    assert result.stored is False
    assert payload["candidate"]["response_fingerprint"] is not None
    assert payload["candidate"]["metadata"]["response_linked"] is True
    assert payload["candidate"]["paamax_metadata"]["memory_store_mutation_by_default"] is False
    assert len(payload["candidate"]["parameter_refs"]) > 0
    json.dumps(payload)


def test_context_memory_builder_can_stage_with_explicit_permission_to_shared_and_qh():
    cfg = ContextCompressionConfig(dim=16, max_parameter_refs=4)
    builder = ContextEpisodicMemoryBuilder(cfg)
    store = SharedSlotStore(SharedSlotStoreConfig(namespace="test_context", dim=16))
    qh = QuantumHolographicStorage(QuantumHolographicStorageConfig(dim=16, num_depths=8), shared_slot_store=store)
    context = torch.randn(2, 6, 16)
    response = torch.randn(2, 5, 16)

    result = builder.build_and_stage(
        context=context,
        response=response,
        shared_slot_store=store,
        qh_storage=qh,
        allow_store=True,
        write_permission=True,
        project_id="qdt-wm-maae",
        chat_id="chat-002",
        episode_id="episode-002",
        context_map="quantum_holographic",
        task_mode="context_episode",
    )

    payload = result.to_dict()
    assert result.stored is True
    assert payload["shared_slot_result"]["canonical_id"].startswith("css-")
    assert payload["qh_record"]["record_id"].startswith("qhrec-")
    assert payload["qh_record"]["code_schema"]["notice"] == "quantum_holographic_compatible_metadata_not_quantum_hardware_claim"
    json.dumps(payload)


def test_geometry_mounted_context_buffer_exposes_context_memory_candidate():
    buffer = GeometryMountedContextBuffer(dim=16, num_depths=8)
    context = torch.randn(2, 5, 16)
    response = torch.randn(2, 4, 16)
    model = TinyContextModel(dim=16)

    candidate, vector, compression = buffer.build_context_memory_candidate(
        context=context,
        response=response,
        model=model,
        project_id="project-a",
        chat_id="chat-a",
        episode_id="ep-a",
        context_map="quantum_holographic",
        task_hints=["context", "episodic"],
    )

    assert vector.shape == (16,)
    assert candidate.context_map == "quantum_holographic"
    assert candidate.response_fingerprint is not None
    assert len(candidate.weight_refs) > 0
    json.dumps(candidate.to_dict())
    json.dumps(compression.to_dict())


def test_context_compression_contract_is_json_safe():
    payload = wm_context_compression_contract()
    assert payload["payload"]["context_compression"] is True
    assert payload["paamax_metadata"]["write_permission_required"] is True
    json.dumps(payload)
