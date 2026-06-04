from mnemonic_cortex.working_memory.qspin_extended_safety_regression import ExtendedSafetyRegressionRunner, ExtendedSafetyCaseKind

def test_extended_safety_regression_complete():
    result = ExtendedSafetyRegressionRunner().run()
    assert result.passed
    kinds = {r.case.kind for r in result.results}
    assert ExtendedSafetyCaseKind.NO_LIVE_ROUTE_ACTIVATION in kinds
    assert ExtendedSafetyCaseKind.NO_PRODUCTION_ACTIVATION in kinds
