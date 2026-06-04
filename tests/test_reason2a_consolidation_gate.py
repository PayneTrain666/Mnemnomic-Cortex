import json

from mnemonic_cortex.reasoning_depth import (
    ShadowConsolidationGate,
    ConsolidationGateConfig,
    ConsolidationDecision,
)


def test_reason2a_consolidation_gate_denies_or_shadows_by_default():
    gate = ShadowConsolidationGate(ConsolidationGateConfig())
    proposal = {"candidate": "memory"}
    result = gate.evaluate(proposal, write_permission=False, confidence=0.9, disagreement=0.0, canonical_slot_id="slot.1")
    payload = result.to_dict()
    assert payload["decision"] == ConsolidationDecision.SHADOW_ONLY.value
    assert payload["committed"] is False
    assert payload["shadow_only"] is True
    assert payload["paamax_metadata"]["write_permission_granted"] is False
    json.dumps(payload)


def test_reason2a_consolidation_gate_quarantines_conflict():
    gate = ShadowConsolidationGate(ConsolidationGateConfig())
    result = gate.evaluate({"candidate": "memory"}, write_permission=True, confidence=0.9, disagreement=0.0, conflict=True)
    assert result.decision == ConsolidationDecision.QUARANTINED
    assert result.committed is False
