from mnemonic_cortex.working_memory.qspin_ci_matrix_hardening import CIMatrixRunner, CIMatrixAxis

def test_ci_matrix_blocks_unsafe_axes_without_failing():
    result = CIMatrixRunner().run()
    assert result.passed
    axes = {r.axis for r in result.results}
    assert CIMatrixAxis.PRODUCTION_ACTIVATION_ATTEMPT in axes
    assert CIMatrixAxis.QH_WRITE_ATTEMPT in axes
