from mnemonic_cortex.working_memory.quality import WMQualityIssue, WMQualityIssueFamily, WMQualitySeverity, WMQualityRemediationPlanner

def test_remediation_plan_is_non_mutating_and_deterministic():
    issue=WMQualityIssue(WMQualityIssueFamily.MISSING_PAAMAX_METADATA,WMQualitySeverity.HIGH,'WM-QD-4A',['mnemonic_cortex/working_memory/wm_shared_slot_store.py'],'missing paamax')
    planner=WMQualityRemediationPlanner(max_plans=4); a=planner.plan_for_issue(issue); b=planner.plan_for_issue(issue)
    assert a.plan_id==b.plan_id; assert a.no_auto_apply is True; assert 'model weight mutation' in ' '.join(a.blocked_unsafe_actions); assert a.to_dict()['safety_payload']['no_auto_apply'] is True

def test_remediation_plan_set_is_bounded():
    issues=[WMQualityIssue(WMQualityIssueFamily.WEAK_TEST_COVERAGE,WMQualitySeverity.MEDIUM,'WM-QD-1A',[f'tests/t{i}.py'],'weak tests') for i in range(3)]
    d=WMQualityRemediationPlanner(max_plans=1).build_plan_set(issues).to_dict(); assert d['plan_count']==1; assert d['truncated'] is True
