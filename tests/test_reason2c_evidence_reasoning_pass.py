import json
import torch

from mnemonic_cortex.reasoning_depth import EvidenceReasoningPass, EvidenceReasoningConfig


def test_reason2c_evidence_report_serializes_and_is_bounded():
    engine = EvidenceReasoningPass(EvidenceReasoningConfig.enabled_default())
    report = engine.run(content="First claim. Second claim. Third claim.", query=torch.randn(2, 3, 16))
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert len(payload["evidence_units"]) <= 12
    assert payload["aggregate_support"] >= 0.0
    assert payload["paamax_metadata"]["evidence_units"] is True
    json.dumps(payload)


def test_reason2c_evidence_disabled_default():
    engine = EvidenceReasoningPass(EvidenceReasoningConfig.disabled())
    report = engine.run(content="No extraction")
    assert report.enabled is False
    assert report.evidence_units == []
