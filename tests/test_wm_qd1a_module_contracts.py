import importlib
import json

EARLY_MODULES = [
    "context_geometry_maps",
    "context_map_selector",
    "context_to_wm_bridge",
    "wm_context_mount",
    "legacy_enhanced_curved_memory",
    "wm_curved_core",
    "curved_resonant_wm_core",
    "curved_slot_state",
    "curvature_metric_policy",
    "geometry_aware_addressing",
    "bounded_associative_spread",
    "curved_local_trace",
    "curved_shadow_write",
]


def test_early_foundation_modules_expose_qd1a_contracts():
    missing = []
    for name in EARLY_MODULES:
        module = importlib.import_module(f"mnemonic_cortex.working_memory.{name}")
        if not hasattr(module, "wm_qd1a_foundation_contract"):
            missing.append(name)
            continue
        contract = module.wm_qd1a_foundation_contract()
        json.dumps(contract)
        assert contract["paamax_metadata"]["trace_governance"] is True
        assert contract["payload"]["shape_checks_required"] is True
        assert contract["payload"]["finite_checks_required"] is True
        assert contract["payload"]["serialization_safe"] is True
    assert not missing
