from mnemonic_cortex.working_memory.qspin_shadow_stress_replay import ShadowStressReplayEngine, StressReplayInput, StressReplayStatus

def test_stress_replay_passes_and_blocks_live_request():
    engine = ShadowStressReplayEngine()
    ok = engine.run(StressReplayInput())
    assert ok.passed
    bad = engine.run(StressReplayInput(live_routing_requested=True))
    assert not bad.passed
    assert bad.fail_count > 0
