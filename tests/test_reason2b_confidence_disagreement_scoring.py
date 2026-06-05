import json
import torch
import pytest

from mnemonic_cortex.reasoning_depth import score_confidence_disagreement, ConfidenceDisagreementConfig, ConfidenceDisagreementError


def test_reason2b_confidence_disagreement_finite_serializable():
    scores = torch.tensor([2.0, 1.0, 0.25])
    report = score_confidence_disagreement(scores, support_mass=0.8)
    payload = report.to_dict()

    assert 0.0 <= payload["confidence"] <= 1.0
    assert 0.0 <= payload["disagreement"] <= 1.0
    assert payload["paamax_metadata"]["confidence_hook"] is True
    json.dumps(payload)


def test_reason2b_confidence_disagreement_rejects_nan():
    with pytest.raises(ConfidenceDisagreementError):
        score_confidence_disagreement(torch.tensor([1.0, float("nan")]), config=ConfidenceDisagreementConfig())
