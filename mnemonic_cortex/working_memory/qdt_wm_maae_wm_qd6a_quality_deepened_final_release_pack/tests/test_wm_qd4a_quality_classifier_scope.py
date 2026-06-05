from pathlib import Path

from mnemonic_cortex.working_memory.quality import WMQualityClassifier, WMQualityClassifierConfig, WMQualityRemediationPlanner


def test_qd4a_classifier_scans_external_memory_scope_bounded():
    root = Path.cwd()
    patterns = [
        "mnemonic_cortex/working_memory/wm_external_memory_interfaces.py",
        "mnemonic_cortex/working_memory/wm_ltm_cross_attention.py",
        "mnemonic_cortex/working_memory/wm_mann_cross_attention.py",
        "mnemonic_cortex/working_memory/wm_spcp_cross_attention.py",
        "mnemonic_cortex/working_memory/wm_dual_fusion.py",
        "mnemonic_cortex/working_memory/wm_shared_slot_registry.py",
        "mnemonic_cortex/working_memory/wm_shared_slot_store.py",
        "mnemonic_cortex/working_memory/wm_quantum_holographic_storage.py",
    ]
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=32, max_issues=128))
    issue_set = clf.classify_tree(root, include_globs=patterns)
    d = issue_set.to_dict()
    assert d["issue_count"] <= 128
    assert d["safety_payload"]["bounded"] is True


def test_qd4a_remediation_planner_routes_external_files_to_qd4a():
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=8, max_issues=32))
    issues = clf.classify_source_text(
        "mnemonic_cortex/working_memory/wm_shared_slot_store.py",
        "import torch\ndef f(x):\n    return x\n",
    )
    plans = WMQualityRemediationPlanner(max_plans=32).build_plan_set(issues).to_dict()
    assert plans["plan_count"] > 0
    assert all(p["owner"] == "WM-QD-4A" for p in plans["plans"])
