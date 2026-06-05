from mnemonic_cortex.working_memory.qspin_shadow_stress_replay import ShadowStressReplayEngine, StressReplayInput
from mnemonic_cortex.working_memory.qspin_trace_corpus import TraceCorpusRecord, TraceCorpusRecordKind

def test_abuse_live_write_commit_production_rejected():
    result = ShadowStressReplayEngine().run(StressReplayInput(write_requested=True, commit_requested=True, production_activation_requested=True))
    assert not result.passed
    assert result.fail_count > 0

def test_trace_corpus_rejects_secrets():
    try:
        TraceCorpusRecord(TraceCorpusRecordKind.REDACTION_CANARY, 'bad', {'secret': 'x'}).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('secret should be rejected')
