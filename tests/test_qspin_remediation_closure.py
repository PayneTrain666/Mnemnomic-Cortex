from mnemonic_cortex.working_memory.qspin_remediation_closure import RemediationClosureWorkflow, RemediationClosureRecord, RemediationClosureStatus

def test_remediation_closure_allows_default_and_blocks_open_p0():
    assert RemediationClosureWorkflow().close().passed
    blocked = RemediationClosureWorkflow().close((RemediationClosureRecord('p0', 'P0', RemediationClosureStatus.OPEN),))
    assert not blocked.passed
