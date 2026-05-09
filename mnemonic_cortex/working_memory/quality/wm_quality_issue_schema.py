from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
import hashlib, json, time

class WMQualitySeverity(str, Enum):
    BLOCKER='blocker'; HIGH='high'; MEDIUM='medium'; LOW='low'; ENHANCEMENT='enhancement'

class WMQualityIssueFamily(str, Enum):
    FAKE_DONE_MODULE='fake_done_module'; SHALLOW_SKELETON='shallow_skeleton'; WEAK_TYPE_CONTRACT='weak_type_contract'
    MISSING_SHAPE_CHECK='missing_shape_check'; MISSING_FINITE_TENSOR_CHECK='missing_finite_tensor_check'
    MISSING_TRACE_HOOK='missing_trace_hook'; MISSING_STABILITY_HOOK='missing_stability_hook'; MISSING_PAAMAX_METADATA='missing_paamax_metadata'
    MISSING_LINEAGE='missing_lineage'; MISSING_SERIALIZATION_SAFETY='missing_serialization_safety'; MISSING_NO_MUTATION_GUARANTEE='missing_no_mutation_guarantee'
    WEAK_FALLBACK_BEHAVIOUR='weak_fallback_behaviour'; WEAK_TEST_COVERAGE='weak_test_coverage'; WEAK_INTEGRATION_CONTRACT='weak_integration_contract'
    DOC_TEST_SOURCE_MISMATCH='doc_test_source_mismatch'; PERFORMANCE_RISK='performance_risk'; UNBOUNDED_SCAN_OR_GROWTH='unbounded_scan_or_growth'
    REAL_SOURCE_INTEGRATION_GAP='real_source_integration_gap'; PRODUCTION_HARDENING_DEFERRAL='production_hardening_deferral'; UNKNOWN_SIGNATURE='unknown_signature'

class WMQualityPatchCategory(str, Enum):
    SOURCE_CONTRACT='source_contract'; TEST_STRENGTHENING='test_strengthening'; DOC_UPDATE='doc_update'; TRACE_METADATA='trace_metadata'
    SERIALIZATION='serialization'; BOUNDEDNESS='boundedness'; FALLBACK='fallback'; INTEGRATION='integration'; PRODUCTION_HARDENING='production_hardening'; NO_ACTION_SAFE_DEFER='no_action_safe_defer'

def stable_quality_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload=json.dumps(parts, sort_keys=True, default=str).encode('utf-8')
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:length]}"

@dataclass(frozen=True)
class WMQualityLineageRef:
    source_pack: str; stage_id: str; file_path: Optional[str]=None; source_hash: Optional[str]=None
    test_path: Optional[str]=None; doc_path: Optional[str]=None; tracker_id: Optional[str]=None; deferred_id: Optional[str]=None
    benchmark_ref: Optional[str]=None; release_manifest_entry: Optional[str]=None
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class WMQualityEvidence:
    evidence_type: str; summary: str; snippet: Optional[str]=None; line_number: Optional[int]=None; metadata: Dict[str, Any]=field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class WMQualityIssue:
    family: WMQualityIssueFamily; severity: WMQualitySeverity; affected_stage: str; affected_files: List[str]; summary: str
    evidence: List[WMQualityEvidence]=field(default_factory=list); lineage: List[WMQualityLineageRef]=field(default_factory=list)
    issue_id: Optional[str]=None; created_at: float=field(default_factory=lambda: time.time()); safety_payload: Dict[str, Any]=field(default_factory=dict); status: str='incomplete'
    def __post_init__(self):
        if self.issue_id is None:
            self.issue_id=stable_quality_id('wmqi', self.family.value, self.severity.value, self.affected_stage, sorted(self.affected_files), self.summary)
        if not self.safety_payload:
            self.safety_payload.update({
                'classification_only': True,
                'no_mutation': True,
                'no_auto_patch_application': True,
                'no_model_weight_mutation': True,
                'no_optimizer_mutation': True,
                'no_memory_store_mutation': True,
            })
        self.validate()
    def validate(self):
        if not self.affected_stage: raise ValueError('affected_stage must be non-empty')
        if not self.affected_files: raise ValueError('affected_files must be non-empty')
        if self.status not in {'incomplete','partially_complete','complete','superseded','rejected'}: raise ValueError('invalid status')
        if not str(self.issue_id).startswith('wmqi-'): raise ValueError('issue_id must start with wmqi-')
    def to_dict(self)->Dict[str,Any]:
        self.validate(); return {'issue_id':self.issue_id,'family':self.family.value,'severity':self.severity.value,'affected_stage':self.affected_stage,'affected_files':list(self.affected_files),'summary':self.summary,'evidence':[e.to_dict() for e in self.evidence],'lineage':[l.to_dict() for l in self.lineage],'created_at':self.created_at,'safety_payload':self.safety_payload,'status':self.status}

@dataclass
class WMQualityIssueSet:
    issues: List[WMQualityIssue]=field(default_factory=list); max_issues: int=256; truncated: bool=False
    def add(self, issue: WMQualityIssue):
        if len(self.issues)>=self.max_issues: self.truncated=True; return
        self.issues.append(issue)
    def extend(self, issues: Iterable[WMQualityIssue]):
        for i in issues: self.add(i)
    def by_severity(self, severity: WMQualitySeverity): return [i for i in self.issues if i.severity==severity]
    def to_dict(self): return {'issue_count':len(self.issues),'max_issues':self.max_issues,'truncated':self.truncated,'issues':[i.to_dict() for i in self.issues],'safety_payload':{'classification_only':True,'no_model_weight_mutation':True,'no_optimizer_mutation':True,'no_memory_store_mutation':True,'bounded':True}}
