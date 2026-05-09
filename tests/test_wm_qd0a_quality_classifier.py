from pathlib import Path
from mnemonic_cortex.working_memory.quality import WMQualityClassifier, WMQualityClassifierConfig, WMQualityIssueFamily

def test_classifier_detects_missing_shape_and_finite_checks():
    text='import torch\ndef forward(x):\n    return x + 1\n'
    issues=WMQualityClassifier(WMQualityClassifierConfig(max_files=4,max_issues=16)).classify_source_text('mnemonic_cortex/working_memory/foo.py',text)
    fam={i.family for i in issues}; assert WMQualityIssueFamily.MISSING_SHAPE_CHECK in fam; assert WMQualityIssueFamily.MISSING_FINITE_TENSOR_CHECK in fam

def test_classifier_detects_pytest_failure():
    issues=WMQualityClassifier().classify_pytest_output('1 failed, 2 passed','docs/qdt_wm_maae/pytest.txt')
    assert issues and issues[0].severity.value=='blocker'

def test_classifier_detects_deferred_real_source_gap():
    issues=WMQualityClassifier().classify_tracker_text('real EnhancedMnemonicCortex source patch pending partially_complete','docs/qdt_wm_maae/08_deferred_work_register.md')
    assert issues and issues[0].family==WMQualityIssueFamily.REAL_SOURCE_INTEGRATION_GAP

def test_classifier_tree_is_bounded(tmp_path: Path):
    src=tmp_path/'mnemonic_cortex'/'working_memory'; src.mkdir(parents=True)
    for i in range(5): (src/f'm{i}.py').write_text('import torch\ndef f(x):\n    return x\n')
    s=WMQualityClassifier(WMQualityClassifierConfig(max_files=2,max_issues=8)).classify_tree(tmp_path, include_globs=['mnemonic_cortex/working_memory/*.py'])
    assert s.truncated is True; assert len(s.issues)<=8
