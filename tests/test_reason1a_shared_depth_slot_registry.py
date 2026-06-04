import json
from mnemonic_cortex.reasoning_depth import SharedDepthSlotRegistry

def test_reason1a_shared_registry_links_wm_mann_ltm_by_canonical_id_not_tensors():
    reg=SharedDepthSlotRegistry(); rec=reg.create_or_update(canonical_slot_id='concept.depth.000042', content='depth indexed reasoning memory', wm_ref='wm.slot.12', mann_ref='mann.slot.88', ltm_ref='ltm.cgmn.2041', project_id='mnemonic_cortex', chat_id='chat.1', episode_id='episode.1', depth_roles_present=[0,1,2,3,5], confidence=0.91, disagreement=0.07)
    assert rec.canonical_slot_id == 'concept.depth.000042'; assert rec.wm_refs == ['wm.slot.12']; assert rec.mann_refs == ['mann.slot.88']; assert rec.ltm_refs == ['ltm.cgmn.2041']; payload=reg.to_dict(); json.dumps(payload); assert payload['safety']['shared_physical_tensor'] is False

def test_reason1a_shared_registry_marks_content_disagreement_without_tensor_coupling():
    reg=SharedDepthSlotRegistry(); reg.create_or_update(canonical_slot_id='concept.x', content='alpha', wm_ref='wm.1'); rec=reg.create_or_update(canonical_slot_id='concept.x', content='beta', mann_ref='mann.1'); assert rec.disagreement >= 0.5; assert 'mann.1' in rec.mann_refs
