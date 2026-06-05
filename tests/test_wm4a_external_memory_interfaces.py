import torch

from mnemonic_cortex.working_memory import (
    ExternalMemoryQuery,
    ExternalMemoryResponse,
    SyntheticExternalMemoryBank,
)


def test_external_memory_query_response_contract_ltm():
    bank = SyntheticExternalMemoryBank("ltm", dim=32, slots=8)
    query = ExternalMemoryQuery("ltm", query_state=torch.randn(2, 32), metadata={"unit": True})
    response = bank.query(query, top_k=3)

    assert response.memory_state.shape == (2, 3, 32)
    assert response.scores.shape == (2, 3)
    assert response.confidence.shape == (2,)
    assert len(response.slot_ids) == 2
    assert len(response.slot_ids[0]) == 3
    assert response.metadata["adapter_kind"] == "synthetic_contract_adapter"


def test_synthetic_mann_response_has_trace_visibility_tensors():
    bank = SyntheticExternalMemoryBank("mann", dim=32, slots=8, hops=3)
    query = ExternalMemoryQuery("mann", query_state=torch.randn(2, 32))
    response = bank.query(query, top_k=4)

    assert response.scratchpad_tokens.shape == (2, 3, 32)
    assert response.per_hop_attention.shape == (2, 3, 4)
    assert torch.allclose(response.per_hop_attention.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
    response.validate()


def test_external_memory_query_rejects_bad_memory_type():
    try:
        ExternalMemoryQuery("bad", query_state=torch.randn(2, 32)).validate()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
