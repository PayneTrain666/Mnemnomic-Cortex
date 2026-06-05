import json
import torch
import pytest

from mnemonic_cortex.working_memory import (
    WMExternalMemoryValidationError,
    ensure_external_memory_response,
    ensure_mann_trace_visibility,
    ensure_fusion_inputs,
    ensure_shared_slot_id,
    ensure_shared_slot_record,
    ensure_qh_code_schema,
    ensure_qh_storage_record,
    interference_score,
    external_memory_trace,
    external_memory_contract_trace,
)


def test_external_memory_response_and_mann_trace_validation():
    response = {
        "output": torch.randn(2, 5, 32),
        "scores": torch.randn(2, 5),
        "confidence": 0.8,
        "disagreement": 0.1,
        "trace": {"ok": True},
    }
    ensure_external_memory_response("response", response, expected_dim=32)

    mann_trace = {
        "pre_fusion_outputs": {"mann": [1]},
        "per_hop_attention": [[0.5, 0.5]],
        "scratchpad_tokens": ["a", "b"],
        "confidence": 0.7,
        "disagreement": 0.2,
    }
    ensure_mann_trace_visibility("mann_trace", mann_trace)

    with pytest.raises(WMExternalMemoryValidationError):
        ensure_mann_trace_visibility("bad_mann_trace", {"confidence": 0.5})


def test_fusion_input_validation_and_interference_score():
    a = torch.randn(2, 5, 32)
    b = torch.randn(2, 5, 32)
    c = torch.randn(2, 5, 32)
    ensure_fusion_inputs(a, b, c, expected_dim=32)

    score = interference_score(a, a.clone())
    assert 0.99 <= score <= 1.0

    with pytest.raises(WMExternalMemoryValidationError):
        ensure_fusion_inputs(a, torch.randn(2, 4, 32), expected_dim=32)


def test_shared_slot_and_qh_record_validation():
    slot_id = ensure_shared_slot_id("slot", "css-abc12345")
    assert slot_id == "css-abc12345"

    shared = {
        "canonical_id": "css-abc12345",
        "owner": "ltm",
        "conflict_state": "clean",
    }
    ensure_shared_slot_record("shared", shared)

    schema = {
        "depth_code": "depth-00",
        "bank_code": "bank-x",
        "geometry_code": "geo-hphase",
        "triplet_code": "triplet-anchor",
        "memory_type_code": "mem-ltm",
        "task_mode_code": "task-qh",
    }
    ensure_qh_code_schema("schema", schema)

    record = {
        "record_id": "qhrec-abc12345",
        "canonical_slot_id": "css-abc12345",
        "code_schema": schema,
        "paamax_metadata": {"write_permission_granted": True},
    }
    ensure_qh_storage_record("record", record)

    with pytest.raises(WMExternalMemoryValidationError):
        ensure_qh_code_schema("bad_schema", {"depth_code": "depth-00"})


def test_external_memory_traces_are_json_safe_and_paamax_compatible():
    trace = external_memory_trace(module="unit", message="ok", memory_type="ltm", payload={"x": torch.randn(2, 2)})
    contract = external_memory_contract_trace(module="unit")
    json.dumps(trace)
    json.dumps(contract)
    assert trace["paamax_metadata"]["external_memory_contract"] is True
    assert contract["payload"]["no_fake_quantum_claim"] is True
