from pathlib import Path

from mnemonic_cortex.working_memory.quality import WMQualityClassifier, WMQualityClassifierConfig, WMQualityRemediationPlanner


def test_qd5a_classifier_scans_commit_cortex_scope_bounded():
    root = Path.cwd()
    patterns = [
        "mnemonic_cortex/working_memory/wm_system_commit_gate.py",
        "mnemonic_cortex/working_memory/wm_compatibility_wrapper.py",
        "mnemonic_cortex/working_memory/wm_cortex_integration.py",
        "mnemonic_cortex/working_memory/qdt_working_memory.py",
    ]
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=16, max_issues=128))
    issue_set = clf.classify_tree(root, include_globs=patterns)
    d = issue_set.to_dict()
    assert d["issue_count"] <= 128
    assert d["safety_payload"]["bounded"] is True


def test_qd5a_remediation_planner_routes_commit_cortex_files_to_qd5a():
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=8, max_issues=32))
    issues = clf.classify_source_text(
        "mnemonic_cortex/working_memory/wm_system_commit_gate.py",
        "import torch\ndef f(x):\n    return x\n",
    )
    plans = WMQualityRemediationPlanner(max_plans=32).build_plan_set(issues).to_dict()
    assert plans["plan_count"] > 0
    assert all(p["owner"] == "WM-QD-5A" for p in plans["plans"])
