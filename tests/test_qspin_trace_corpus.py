from mnemonic_cortex.working_memory.qspin_trace_corpus import DeterministicTraceCorpusBuilder, TraceCorpusRecordKind

def test_trace_corpus_deterministic_and_complete():
    b = DeterministicTraceCorpusBuilder()
    c1 = b.build(); c2 = b.build()
    assert c1.corpus_hash == c2.corpus_hash
    assert all(c1.coverage[k.value] for k in TraceCorpusRecordKind)
    assert b.replay(c1).passed
