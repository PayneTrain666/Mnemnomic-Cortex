import importlib
import json

ATTENTION_MODULES = [
    "wm_retrieval_lanes",
    "wm_geometry_scoring",
    "wm_memory_augmented_attention",
    "wm_geometry_linker",
    "wm_evidence_attention",
    "wm_trace_attention",
    "wm_counterfactual_attention",
    "wm_conflict_attention",
    "wm_novelty_attention",
    "wm_stability_attention",
    "wm_inter_manifold_attention",
]


def test_attention_modules_expose_qd3a_contracts():
    missing = []
    for name in ATTENTION_MODULES:
        module = importlib.import_module(f"mnemonic_cortex.working_memory.{name}")
        if not hasattr(module, "wm_qd3a_attention_contract"):
            missing.append(name)
            continue
        contract = module.wm_qd3a_attention_contract()
        json.dumps(contract)
        assert contract["trace_type"] == "wm_qd3a_attention_trace"
        assert contract["payload"]["candidate_schema_required"] is True
        assert contract["payload"]["bounded_topk_required"] is True
        assert contract["payload"]["paamax_metadata_required"] is True
        assert contract["paamax_metadata"]["trace_governance"] is True
    assert not missing
