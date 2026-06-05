from pathlib import Path

from mnemonic_cortex.working_memory.quality import WMQualityClassifier, WMQualityClassifierConfig, WMQualityRemediationPlanner


def test_qd2a_classifier_scans_depth_scope_bounded():
    root = Path.cwd()
    patterns = [
        "mnemonic_cortex/working_memory/wm_quaternion_depth.py",
        "mnemonic_cortex/working_memory/wm_intra_depth_transformer.py",
        "mnemonic_cortex/working_memory/wm_cross_depth_transformer.py",
        "mnemonic_cortex/working_memory/depth_specific_addressing.py",
        "mnemonic_cortex/working_memory/wm_depth_adapters.py",
        "mnemonic_cortex/working_memory/wm_depth_fusion.py",
        "mnemonic_cortex/working_memory/wm_triplet_state.py",
        "mnemonic_cortex/working_memory/wm_trace.py",
        "mnemonic_cortex/working_memory/qdt_working_memory.py",
    ]
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=32, max_issues=128))
    issue_set = clf.classify_tree(root, include_globs=patterns)
    d = issue_set.to_dict()
    assert d["issue_count"] <= 128
    assert d["safety_payload"]["bounded"] is True


def test_qd2a_remediation_planner_routes_depth_files_to_qd2a():
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=8, max_issues=32))
    issues = clf.classify_source_text(
        "mnemonic_cortex/working_memory/wm_quaternion_depth.py",
        "import torch\ndef f(x):\n    return x\n",
    )
    plans = WMQualityRemediationPlanner(max_plans=32).build_plan_set(issues).to_dict()
    assert plans["plan_count"] > 0
    assert all(p["owner"] == "WM-QD-2A" for p in plans["plans"])
