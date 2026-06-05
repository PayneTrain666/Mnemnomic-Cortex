import importlib
import json

EXTERNAL_MODULES = [
    "wm_external_memory_interfaces",
    "wm_ltm_cross_attention",
    "wm_mann_cross_attention",
    "wm_spcp_cross_attention",
    "wm_dual_fusion",
    "wm_shared_slot_registry",
    "wm_shared_slot_store",
    "wm_quantum_holographic_storage",
]


def test_external_memory_modules_expose_qd4a_contracts():
    missing = []
    for name in EXTERNAL_MODULES:
        module = importlib.import_module(f"mnemonic_cortex.working_memory.{name}")
        if not hasattr(module, "wm_qd4a_external_memory_contract"):
            missing.append(name)
            continue
        contract = module.wm_qd4a_external_memory_contract()
        json.dumps(contract)
        assert contract["trace_type"] == "wm_qd4a_external_memory_trace"
        assert contract["payload"]["external_response_schema_required"] is True
        assert contract["payload"]["mann_trace_visibility_required"] is True
        assert contract["payload"]["shared_slot_metadata_required"] is True
        assert contract["payload"]["qh_code_schema_required"] is True
        assert contract["paamax_metadata"]["trace_governance"] is True
    assert not missing
