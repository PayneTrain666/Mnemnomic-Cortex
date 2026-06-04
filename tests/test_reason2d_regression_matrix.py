import json

from mnemonic_cortex.reasoning_depth import build_reasoning_regression_matrix


def test_reason2d_regression_matrix_serializes_and_tracks_deferred_work():
    matrix = build_reasoning_regression_matrix()
    payload = matrix.to_dict()

    assert payload["summary"]["total_rows"] >= 10
    assert any(row["stage"] == "REASON-3A" and row["deferred_work"] for row in payload["rows"])
    json.dumps(payload)
