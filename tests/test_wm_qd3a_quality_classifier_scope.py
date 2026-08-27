from pathlib import Path

from mnemonic_cortex.working_memory.quality import WMQualityClassifier, WMQualityClassifierConfig, WMQualityRemediationPlanner


def test_qd3a_classifier_scans_attention_scope_bounded():
    root = Path.cwd()
    patterns = [
        "mnemonic_cortex/working_memory/wm_retrieval_lanes.py",
        "mnemonic_cortex/working_memory/wm_geometry_scoring.py",
        "mnemonic_cortex/working_memory/wm_memory_augmented_attention.py",
        "mnemonic_cortex/working_memory/wm_geometry_linker.py",
        "mnemonic_cortex/working_memory/wm_evidence_attention.py",
        "mnemonic_cortex/working_memory/wm_trace_attention.py",
        "mnemonic_cortex/working_memory/wm_counterfactual_attention.py",
        "mnemonic_cortex/working_memory/wm_conflict_attention.py",
        "mnemonic_cortex/working_memory/wm_novelty_attention.py",
        "mnemonic_cortex/working_memory/wm_stability_attention.py",
        "mnemonic_cortex/working_memory/wm_inter_manifold_attention.py",
    ]
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=32, max_issues=128))
    issue_set = clf.classify_tree(root, include_globs=patterns)
    d = issue_set.to_dict()
    assert d["issue_count"] <= 128
    assert d["safety_payload"]["bounded"] is True


def test_qd3a_remediation_planner_routes_attention_files_to_qd3a():
    clf = WMQualityClassifier(WMQualityClassifierConfig(max_files=8, max_issues=32))
    issues = clf.classify_source_text(
        "mnemonic_cortex/working_memory/wm_memory_augmented_attention.py",
        "import torch\ndef f(x):\n    return x\n",
    )
    plans = WMQualityRemediationPlanner(max_plans=32).build_plan_set(issues).to_dict()
    assert plans["plan_count"] > 0
    assert all(p["owner"] == "WM-QD-3A" for p in plans["plans"])
