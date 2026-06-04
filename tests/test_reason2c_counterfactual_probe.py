import json

from mnemonic_cortex.reasoning_depth import (
    EvidenceReasoningPass,
    EvidenceReasoningConfig,
    CounterfactualReasoningProbe,
    CounterfactualProbeConfig,
)


def test_reason2c_counterfactual_probe_bounded_and_serializable():
    evidence = EvidenceReasoningPass(EvidenceReasoningConfig(enabled=True, max_evidence_units=3)).run(
        content="A. B. C. D."
    )
    probe = CounterfactualReasoningProbe(CounterfactualProbeConfig(enabled=True, max_probes=2))
    report = probe.run(evidence, base_confidence=0.8, base_disagreement=0.1)
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert len(payload["probes"]) <= 2
    assert payload["safety"]["non_mutating"] is True
    json.dumps(payload)
