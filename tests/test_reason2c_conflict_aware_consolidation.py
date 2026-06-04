import json

from mnemonic_cortex.reasoning_depth import (
    EvidenceReasoningPass,
    EvidenceReasoningConfig,
    CounterfactualReasoningProbe,
    CounterfactualProbeConfig,
    ConflictAwareConsolidationEvaluator,
    ConflictAwareConsolidationConfig,
)


def test_reason2c_conflict_aware_consolidation_quarantine_metadata():
    evidence = EvidenceReasoningPass(EvidenceReasoningConfig(enabled=True, max_evidence_units=1)).run(content="weak")
    counterfactual = CounterfactualReasoningProbe(CounterfactualProbeConfig(enabled=True)).run(evidence)
    evaluator = ConflictAwareConsolidationEvaluator(ConflictAwareConsolidationConfig.enabled_default())
    report = evaluator.evaluate(
        evidence_report=evidence,
        counterfactual_report=counterfactual,
        confidence=0.2,
        disagreement=0.8,
        base_conflict=True,
    )
    payload = report.to_dict()

    assert payload["enabled"] is True
    assert payload["conflict_detected"] is True
    assert payload["quarantine_recommended"] is True
    assert payload["paamax_metadata"]["quarantine_hook"] is True
    json.dumps(payload)
