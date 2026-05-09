import importlib
import json

COMMIT_CORTEX_MODULES = [
    "wm_system_commit_gate",
    "wm_compatibility_wrapper",
    "wm_cortex_integration",
    "qdt_working_memory",
]


def test_commit_cortex_modules_expose_qd5a_contracts():
    missing = []
    for name in COMMIT_CORTEX_MODULES:
        module = importlib.import_module(f"mnemonic_cortex.working_memory.{name}")
        if not hasattr(module, "wm_qd5a_commit_cortex_contract"):
            missing.append(name)
            continue
        contract = module.wm_qd5a_commit_cortex_contract()
        json.dumps(contract)
        assert contract["trace_type"] == "wm_qd5a_commit_cortex_trace"
        assert contract["payload"]["commit_proposal_schema_required"] is True
        assert contract["payload"]["rollback_trace_safety_required"] is True
        assert contract["payload"]["cortex_migration_template_safety_required"] is True
        assert contract["payload"]["no_fake_real_source_patch_claim"] is True
        assert contract["paamax_metadata"]["trace_governance"] is True
    assert not missing
