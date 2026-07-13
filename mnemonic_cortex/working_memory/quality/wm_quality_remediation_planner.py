"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: wm quality remediation planner.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List
import time
from .wm_quality_issue_schema import WMQualityIssue, WMQualityIssueFamily, WMQualityPatchCategory, stable_quality_id

class WMRemediationOwner(str, Enum):
    WM_QD_1A='WM-QD-1A'; WM_QD_2A='WM-QD-2A'; WM_QD_3A='WM-QD-3A'; WM_QD_4A='WM-QD-4A'; WM_QD_5A='WM-QD-5A'; WM_QD_6A='WM-QD-6A'; PRODUCTION_HARDENING='production_hardening'; REAL_SOURCE_INTEGRATION='real_source_integration'

@dataclass
class WMRemediationPlan:
    plan_id: str; issue_id: str; owner: WMRemediationOwner; patch_category: WMQualityPatchCategory; recommended_action: str; required_evidence: List[str]; blocked_unsafe_actions: List[str]
    no_auto_apply: bool=True; lineage_preserved: bool=True; bounded: bool=True; created_at: float=field(default_factory=lambda: time.time()); metadata: Dict[str, Any]=field(default_factory=dict)
    def validate(self):
        if not self.plan_id.startswith('wmrp-'): raise ValueError('plan_id must start with wmrp-')
        if not self.issue_id.startswith('wmqi-'): raise ValueError('issue_id must start with wmqi-')
        if not self.no_auto_apply or not self.bounded: raise ValueError('plans must be no-auto-apply and bounded')
    def to_dict(self):
        self.validate(); out=asdict(self); out['owner']=self.owner.value; out['patch_category']=self.patch_category.value; out['safety_payload']={'remediation_planning_only':True,'no_auto_apply':True,'no_model_weight_mutation':True,'no_optimizer_mutation':True,'no_memory_store_mutation':True,'no_policy_activation':True}; return out

@dataclass
class WMRemediationPlanSet:
    plans: List[WMRemediationPlan]=field(default_factory=list); max_plans: int=256; truncated: bool=False
    def add(self, plan: WMRemediationPlan):
        if len(self.plans)>=self.max_plans: self.truncated=True; return
        plan.validate(); self.plans.append(plan)
    def to_dict(self): return {'plan_count':len(self.plans),'max_plans':self.max_plans,'truncated':self.truncated,'plans':[p.to_dict() for p in self.plans]}

def _owner(path: str, family: WMQualityIssueFamily):
    if family==WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP: return WMRemediationOwner.REAL_SOURCE_INTEGRATION
    if family==WMQualityIssueFamily.PRODUCTION_HARDENING_DEFERRAL: return WMRemediationOwner.PRODUCTION_HARDENING
    if any(n in path for n in ['context_','curved_','legacy_enhanced','geometry_aware','bounded_associative','curvature_metric']): return WMRemediationOwner.WM_QD_1A
    if any(n in path for n in ['quaternion','depth','triplet','qdt_working_memory','wm_trace']): return WMRemediationOwner.WM_QD_2A
    if any(n in path for n in ['retrieval','scoring','attention','novelty','conflict','stability']): return WMRemediationOwner.WM_QD_3A
    if any(n in path for n in ['external_memory','ltm','mann','spcp','dual_fusion','shared_slot','quantum_holographic']): return WMRemediationOwner.WM_QD_4A
    if any(n in path for n in ['commit_gate','compatibility','cortex_integration']): return WMRemediationOwner.WM_QD_5A
    return WMRemediationOwner.WM_QD_6A

def _category(fam):
    return {WMQualityIssueFamily.MISSING_SHAPE_CHECK:WMQualityPatchCategory.SOURCE_CONTRACT,WMQualityIssueFamily.MISSING_FINITE_TENSOR_CHECK:WMQualityPatchCategory.SOURCE_CONTRACT,WMQualityIssueFamily.MISSING_TRACE_HOOK:WMQualityPatchCategory.TRACE_METADATA,WMQualityIssueFamily.MISSING_PAAMAX_METADATA:WMQualityPatchCategory.TRACE_METADATA,WMQualityIssueFamily.UNBOUNDED_SCAN_OR_GROWTH:WMQualityPatchCategory.BOUNDEDNESS,WMQualityIssueFamily.WEAK_TEST_COVERAGE:WMQualityPatchCategory.TEST_STRENGTHENING,WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP:WMQualityPatchCategory.INTEGRATION,WMQualityIssueFamily.PRODUCTION_HARDENING_DEFERRAL:WMQualityPatchCategory.PRODUCTION_HARDENING}.get(fam, WMQualityPatchCategory.SOURCE_CONTRACT)

class WMQualityRemediationPlanner:
    def __init__(self, max_plans:int=256):
        if max_plans<=0: raise ValueError('max_plans must be positive')
        self.max_plans=max_plans
    def plan_for_issue(self, issue: WMQualityIssue):
        primary=issue.affected_files[0]; owner=_owner(primary, issue.family); cat=_category(issue.family)
        action=self._action(issue,cat); pid=stable_quality_id('wmrp',issue.issue_id,owner.value,cat.value,action)
        return WMRemediationPlan(pid, issue.issue_id, owner, cat, action, ['updated source diff or explicit safe deferral','unit/regression test covering the issue','tracker entry updated','ship-check confirms no unsafe mutation'], ['automatic global patching','model weight mutation','optimizer mutation','automatic memory-store mutation','real external adapter activation','policy activation','fake production-complete claim'], metadata={'affected_stage':issue.affected_stage,'affected_files':issue.affected_files,'severity':issue.severity.value,'family':issue.family.value})
    def _action(self, issue, cat):
        fam=issue.family
        if fam==WMQualityIssueFamily.MISSING_SHAPE_CHECK: return 'Add explicit shape validation and invalid-shape tests.'
        if fam==WMQualityIssueFamily.MISSING_FINITE_TENSOR_CHECK: return 'Add finite/NaN/Inf validation and non-finite tests.'
        if fam==WMQualityIssueFamily.MISSING_TRACE_HOOK: return 'Add trace serialization with to_dict-compatible payloads.'
        if fam==WMQualityIssueFamily.MISSING_PAAMAX_METADATA: return 'Add PAAMA-X metadata including permission/conflict/confidence fields.'
        if fam==WMQualityIssueFamily.UNBOUNDED_SCAN_OR_GROWTH: return 'Add max_records/top_k/limit caps and boundedness tests.'
        if fam==WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP: return 'Do not fake patch; apply migration template only when real EnhancedMnemonicCortex source is supplied.'
        if fam==WMQualityIssueFamily.PRODUCTION_HARDENING_DEFERRAL: return 'Keep explicit production-hardening deferral with owner and evidence.'
        if fam==WMQualityIssueFamily.WEAK_TEST_COVERAGE: return 'Add targeted regression tests and rerun full suite.'
        return f'Patch category {cat.value}: strengthen contract, fallback, lineage, and tests.'
    def build_plan_set(self, issues: Iterable[WMQualityIssue]):
        ps=WMRemediationPlanSet(max_plans=self.max_plans)
        for i in issues: ps.add(self.plan_for_issue(i))
        return ps
