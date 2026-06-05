import json
import torch

from mnemonic_cortex.reasoning_depth import LTMDepthAdapter, LTMDepthAdapterConfig


def test_reason1d_ltm_shadow_consolidation_proposal_is_not_committed():
    adapter = LTMDepthAdapter(LTMDepthAdapterConfig.enabled_default(key_dim=16, value_dim=16, slot_count=12))
    result = adapter.propose_consolidation(
        canonical_slot_id="concept.consolidate.1",
        content="candidate content",
        bank_name="cgmn_semantic",
        slot_index=3,
        depth_index=1,
        key=torch.randn(16),
        value=torch.randn(16),
        source_stage="REASON-1D",
        source_pack="/tmp/reason1d.zip",
        project_id="mnemonic",
        chat_id="chat.1",
        episode_id="ep.1",
    )

    assert result["committed"] is False
    assert result["shadow_only"] is True
    assert result["bank_proposal"]["committed"] is False
    assert result["registry_proposal"]["shadow_only"] is True
    assert result["registry_record"]["consolidation_status"] == "shadow_proposed"
    assert result["safety"]["permanent_consolidation_requires_gate"] is True
    assert result["paamax_metadata"]["write_permission_granted"] is False
    json.dumps(result)
