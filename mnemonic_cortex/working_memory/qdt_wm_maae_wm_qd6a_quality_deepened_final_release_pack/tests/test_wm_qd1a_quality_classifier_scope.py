from pathlib import Path

from mnemonic_cortex.working_memory.quality import (
    WMQualityClassifier,
    WMQualityClassifierConfig,
    WMQualityIssueFamily,
    WMQualityRemediationPlanner,
)


def test_qd1a_classifier_can_scan_early_scope_without_unbounded_growth():
    root = Path.cwd()
    patterns = [
        "mnemonic_cortex/working_memory/context_geometry_maps.py",
        "mnemonic_cortex/working_memory/context_map_selector.py",
        "mnemonic_cortex/working_memory/context_to_wm_bridge.py",
        "mnemonic_cortex/working_memory/wm_context_mount.py",
        "mnemonic_cortex/working_memory/curved_*.py",
        "mnemonic_cortex/working_memory/geometry_aware_addressing.py",
        "mnemonic_cortex/working_memory/bounded_associative_spread.py",
    ]
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=64, max_issues=128))
    issue_set = clf.classify_tree(root, include_globs=patterns)
    assert issue_set.to_dict()["issue_count"] <= 128
    assert issue_set.to_dict()["safety_payload"]["no_memory_store_mutation"] is True


def test_qd1a_remediation_planner_routes_early_files_to_qd1a():
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=8, max_issues=32))
    issues = clf.classify_source_text(
        "mnemonic_cortex/working_memory/curved_shadow_write.py",
        "import torch\ndef f(x):\n    return x\n",
    )
    plans = WMQualityRemediationPlanner(max_plans=32).build_plan_set(issues).to_dict()
    assert plans["plan_count"] > 0
    assert all(p["owner"] == "WM-QD-1A" for p in plans["plans"])
    assert all(p["safety_payload"]["no_auto_apply"] is True for p in plans["plans"])
