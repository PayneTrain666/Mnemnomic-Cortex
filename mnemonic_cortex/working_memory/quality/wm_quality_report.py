from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict
import json, time
from .wm_quality_classifier import WMQualityClassifier, WMQualityClassifierConfig
from .wm_quality_lineage import build_lineage_index
from .wm_quality_remediation_planner import WMQualityRemediationPlanner

@dataclass
class WMQualityReport:
    report_id: str; source_root: str; source_pack: str; issue_set: Dict[str,Any]; remediation_plan_set: Dict[str,Any]; lineage_index: Dict[str,Any]; created_at: float=field(default_factory=lambda: time.time()); safety_payload: Dict[str,Any]=field(default_factory=dict)
    def to_dict(self): return asdict(self)
    def write_json(self, path: Path): path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

def build_quality_report(root: Path, source_pack: str, max_files:int=128, max_issues:int=256, max_plans:int=256):
    lineage=build_lineage_index(root, source_pack, max_files=max_files*4)
    clf=WMQualityClassifier(WMQualityClassifierConfig(max_files=max_files,max_issues=max_issues), lineage)
    issue_set=clf.classify_tree(root)
    for rel in ['docs/qdt_wm_maae/102_wm7a_pytest_output.txt','docs/qdt_wm_maae/103_wm7a_benchmark_stdout.txt','docs/qdt_wm_maae/00_patch_upgrade_tracker.md','docs/qdt_wm_maae/08_deferred_work_register.md']:
        p=root/rel
        if p.exists():
            txt=p.read_text(encoding='utf-8', errors='replace')
            if 'pytest' in rel: issue_set.extend(clf.classify_pytest_output(txt, rel))
            elif 'benchmark' in rel: issue_set.extend(clf.classify_benchmark_output(txt, rel))
            else: issue_set.extend(clf.classify_tracker_text(txt, rel))
    plans=WMQualityRemediationPlanner(max_plans=max_plans).build_plan_set(issue_set.issues)
    return WMQualityReport('wm-qdrop-0a', str(root), source_pack, issue_set.to_dict(), plans.to_dict(), lineage.to_dict(), safety_payload={'classification_only':True,'remediation_planning_only':True,'no_auto_runtime_patch':True,'no_model_weight_mutation':True,'no_optimizer_mutation':True,'no_memory_store_mutation':True,'bounded':True})
