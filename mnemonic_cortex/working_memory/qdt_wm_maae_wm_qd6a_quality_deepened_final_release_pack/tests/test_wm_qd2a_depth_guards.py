import json
import torch
import pytest

from mnemonic_cortex.working_memory import (
    WMDepthValidationError,
    ensure_token_state,
    ensure_depth_state,
    ensure_triplet_axis,
    normalize_quaternion,
    ensure_unit_quaternion,
    ensure_quaternion_pack,
    depth_summary,
    token_summary,
    depth_contract_trace,
    assert_depth_compatible_tokens,
)


def test_depth_state_and_token_state_validation():
    depth = torch.randn(2, 8, 5, 3, 32)
    tokens = torch.randn(2, 5, 32)
    assert ensure_depth_state("depth", depth, expected_depths=8, expected_dim=32) is depth
    assert ensure_token_state("tokens", tokens, expected_dim=32) is tokens
    assert_depth_compatible_tokens(depth, tokens)

    bad_triplet = torch.randn(2, 8, 5, 2, 32)
    with pytest.raises(WMDepthValidationError):
        ensure_depth_state("bad_triplet", bad_triplet)

    bad_tokens = torch.randn(2, 4, 32)
    with pytest.raises(WMDepthValidationError):
        assert_depth_compatible_tokens(depth, bad_tokens)


def test_depth_state_rejects_nonfinite():
    depth = torch.randn(2, 8, 5, 3, 32)
    depth[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(WMDepthValidationError):
        ensure_depth_state("depth", depth)


def test_quaternion_normalization_and_pack_validation():
    q = torch.randn(8, 4)
    unit = normalize_quaternion(q)
    ensure_unit_quaternion("unit", unit)
    ensure_quaternion_pack("unit", unit, expected_depths=8)

    with pytest.raises(WMDepthValidationError):
        ensure_unit_quaternion("raw", q)


def test_depth_and_token_summaries_are_json_safe():
    depth = torch.randn(2, 8, 5, 3, 32)
    tokens = torch.randn(2, 5, 32)
    ds = depth_summary(depth)
    ts = token_summary(tokens)
    json.dumps(ds)
    json.dumps(ts)
    assert ds["shape"] == [2, 8, 5, 3, 32]
    assert ts["shape"] == [2, 5, 32]


def test_depth_contract_trace_is_paamax_compatible():
    trace = depth_contract_trace(module="unit")
    json.dumps(trace)
    assert trace["payload"]["depth_state_shape"] == "[B,Z,T,3,D]"
    assert trace["payload"]["quaternion_normalization_required"] is True
    assert trace["paamax_metadata"]["depth_contract"] is True
