from tests.qspin_prod4_module_loader import load_prod4_modules
m=load_prod4_modules()["qspin_runtime_safety_regression"]

def test_default_suite_has_all_cases_and_passes():
    suite=m.build_default_qspin_runtime_safety_regression_suite()
    assert {c.kind for c in suite.cases} == set(m.QSpinRuntimeSafetyRegressionCaseKind)
    result=suite.run()
    assert result.failed_count == 0
    assert result.status is m.QSpinRuntimeSafetyRegressionStatus.PASSED

def test_failed_case_blocks_suite():
    c=m.QSpinRuntimeSafetyRegressionCase("x",m.QSpinRuntimeSafetyRegressionCaseKind.NO_MUTATION_GUARANTEE,"no_mutation",should_pass=False)
    result=m.QSpinRuntimeSafetyRegressionSuite((c,)).run()
    assert result.failed_count == 1
    assert result.status is m.QSpinRuntimeSafetyRegressionStatus.FAILED
