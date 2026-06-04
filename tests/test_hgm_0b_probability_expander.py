import math

from mnemonic_cortex.hypergraph_manifold import (
    MutationDirection,
    MutationToken,
    P_VM,
    P_VMD,
    ProbabilityNormalizationMode,
    TraceRecord,
    infer_probability_shape_contract,
    validate_probability_payload,
    normalize_probability_payload,
    build_probability_expansion,
    expand_from_mutation_tokens,
    extract_top_k_scenarios,
)


def row_sums(payload):
    return [sum(row) for row in payload]


def test_valid_vm_payload_infers_correct_shape():
    contract, result = infer_probability_shape_contract([[0.2, 0.8], [0.5, 0.5]])
    assert result.ok
    assert contract == P_VM


def test_valid_vmd_payload_infers_correct_shape():
    contract, result = infer_probability_shape_contract([[[0.2, 0.8], [0.5, 0.5]]])
    assert result.ok
    assert contract == P_VMD


def test_invalid_negative_probability_fails_validation():
    result = validate_probability_payload([[0.2, -0.1]], P_VM)
    assert not result.ok
    assert any(m.code == "probability_payload.negative" for m in result.errors)


def test_invalid_nan_probability_fails_validation():
    result = validate_probability_payload([[0.2, float("nan")]], P_VM)
    assert not result.ok
    assert any(m.code == "probability_payload.non_finite" for m in result.errors)


def test_row_stochastic_normalization_creates_row_sums_of_one():
    normalized, report = normalize_probability_payload([[2.0, 2.0], [1.0, 3.0]], ProbabilityNormalizationMode.ROW_STOCHASTIC, P_VM)
    assert not report.errors
    assert all(abs(total - 1.0) < 1e-9 for total in row_sums(normalized))
    # idempotent on already row-normalized payload
    normalized2, report2 = normalize_probability_payload(normalized, ProbabilityNormalizationMode.ROW_STOCHASTIC, P_VM)
    assert normalized2 == normalized
    assert not report2.errors


def test_global_sum_normalization_creates_total_sum_of_one():
    normalized, report = normalize_probability_payload([[2.0, 2.0], [1.0, 3.0]], ProbabilityNormalizationMode.GLOBAL_SUM, P_VM)
    total = sum(sum(row) for row in normalized)
    assert abs(total - 1.0) < 1e-9
    assert not report.errors


def test_softmax_normalization_produces_finite_non_negative_rows():
    normalized, report = normalize_probability_payload([[2.0, -2.0], [0.0, 0.0]], ProbabilityNormalizationMode.SOFTMAX, P_VM)
    assert not report.errors
    for row in normalized:
        assert all(math.isfinite(value) and value >= 0.0 for value in row)
        assert abs(sum(row) - 1.0) < 1e-9


def test_unsupported_rank_fails_closed():
    contract, result = infer_probability_shape_contract([1.0, 2.0, 3.0])
    assert contract is None
    assert not result.ok
    assert any(m.code == "probability_shape.unsupported_rank" for m in result.errors)


def test_mutation_tokens_expand_into_vm_matrix():
    tokens = [
        MutationToken("t1", "grip", MutationDirection.POSITIVE, "up", 0.7),
        MutationToken("t2", "grip", MutationDirection.NEGATIVE, "down", 0.3),
        MutationToken("t3", "slip", MutationDirection.POSITIVE, "up", 0.2),
    ]
    result = expand_from_mutation_tokens(tokens)
    assert result.validation.ok
    assert result.contract == P_VM
    assert result.payload == [[0.7, 0.3], [0.2, 0.0]]
    assert result.metadata["variable_ids"] == ("grip", "slip")
    assert result.metadata["magnitude_bin_ids"] == ("up", "down")


def test_build_probability_expansion_validates_and_normalizes_tokens():
    tokens = [
        MutationToken("t1", "grip", MutationDirection.POSITIVE, "up", 2.0),
        MutationToken("t2", "grip", MutationDirection.NEGATIVE, "down", 2.0),
    ]
    # token validation rejects >1 probabilities in direct token expansion by design
    result = build_probability_expansion(tokens, normalization_mode=ProbabilityNormalizationMode.ROW_STOCHASTIC)
    assert not result.validation.ok
    assert any(m.code == "mutation_token.invalid_probability" for m in result.validation.errors)


def test_top_k_extraction_returns_deterministic_ordered_candidates():
    payload = [[0.5, 0.5], [0.1, 0.9]]
    result = extract_top_k_scenarios(payload, 3, P_VM, variable_ids=("v0", "v1"), magnitude_bin_ids=("m0", "m1"))
    assert result.validation.ok
    assert [c.source_indices for c in result.candidates] == [(1, 1), (0, 0), (0, 1)]
    assert result.candidates[0].variable_id == "v1"
    assert result.candidates[0].magnitude_bin_id == "m1"


def test_k_le_zero_returns_empty_result_with_warning():
    result = extract_top_k_scenarios([[0.5, 0.5]], 0, P_VM)
    assert result.validation.ok
    assert result.candidates == tuple()
    assert any(m.code == "scenario_extraction.non_positive_k" for m in result.validation.warnings)


def test_trace_records_are_generated_and_redacted():
    expansion = build_probability_expansion([[2.0, 2.0]], normalization_mode=ProbabilityNormalizationMode.ROW_STOCHASTIC)
    assert expansion.trace_records
    assert expansion.validation.ok
    trace = TraceRecord.create("create", "test", payload={"api_key": "abc", "safe": "ok"})
    assert trace.redacted_payload()["api_key"] == "<redacted>"
    assert trace.redacted_payload()["safe"] == "ok"
