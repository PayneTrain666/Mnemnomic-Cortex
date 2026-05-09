from mnemonic_cortex.working_memory.quality import WMQualityIssue, WMQualityIssueFamily, WMQualitySeverity, WMQualityEvidence, WMQualityLineageRef, WMQualityIssueSet, stable_quality_id

def test_quality_issue_deterministic_id_and_serialization():
    a=WMQualityIssue(WMQualityIssueFamily.MISSING_SHAPE_CHECK,WMQualitySeverity.HIGH,'WM-QD-1A',['mnemonic_cortex/working_memory/context_geometry_maps.py'],'shape check missing',evidence=[WMQualityEvidence('unit','summary')],lineage=[WMQualityLineageRef('pack.zip','WM-QD-1A',file_path='x.py')])
    b=WMQualityIssue(WMQualityIssueFamily.MISSING_SHAPE_CHECK,WMQualitySeverity.HIGH,'WM-QD-1A',['mnemonic_cortex/working_memory/context_geometry_maps.py'],'shape check missing')
    assert a.issue_id==b.issue_id
    d=a.to_dict(); assert d['issue_id'].startswith('wmqi-'); assert d['family']=='missing_shape_check'; assert d['safety_payload']['classification_only'] is True

def test_quality_issue_set_is_bounded():
    s=WMQualityIssueSet(max_issues=1); i=WMQualityIssue(WMQualityIssueFamily.WEAK_TEST_COVERAGE,WMQualitySeverity.MEDIUM,'WM-QD-0A',['tests/x.py'],'missing test')
    s.add(i); s.add(i); d=s.to_dict(); assert d['issue_count']==1; assert d['truncated'] is True

def test_stable_quality_id_is_stable():
    assert stable_quality_id('x','a',1)==stable_quality_id('x','a',1)
    assert stable_quality_id('x','a',1)!=stable_quality_id('x','a',2)
