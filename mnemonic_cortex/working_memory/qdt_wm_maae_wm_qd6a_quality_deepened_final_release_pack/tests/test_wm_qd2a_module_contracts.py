import importlib
import json

DEPTH_MODULES = [
    "wm_quaternion_depth",
    "wm_intra_depth_transformer",
    "wm_cross_depth_transformer",
    "depth_specific_addressing",
    "wm_depth_adapters",
    "wm_depth_fusion",
    "wm_triplet_state",
    "wm_trace",
    "qdt_working_memory",
]


def test_depth_modules_expose_qd2a_contracts():
    missing = []
    for name in DEPTH_MODULES:
        module = importlib.import_module(f"mnemonic_cortex.working_memory.{name}")
        if not hasattr(module, "wm_qd2a_depth_contract"):
            missing.append(name)
            continue
        contract = module.wm_qd2a_depth_contract()
        json.dumps(contract)
        assert contract["trace_type"] == "wm_qd2a_depth_contract"
        assert contract["payload"]["depth_state_shape"] == "[B,Z,T,3,D]"
        assert contract["payload"]["triplet_size"] == 3
        assert contract["payload"]["quaternion_normalization_required"] is True
        assert contract["paamax_metadata"]["trace_governance"] is True
    assert not missing
