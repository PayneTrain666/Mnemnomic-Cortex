from tests.qspin_prod4_module_loader import load_prod4_modules
m=load_prod4_modules()["qspin_synthetic_payload_harness"]

def env(kind=m.QSpinSyntheticPayloadKind.DENSE):
    return m.QSpinSyntheticPayloadEnvelope(kind,"src","dst",m.QSpinSyntheticPayloadShape((2,4)),m.QSpinSyntheticPayloadBudget(64),lifecycle=(m.QSpinSyntheticPayloadLifecycle.PLANNED,m.QSpinSyntheticPayloadLifecycle.GENERATED_METADATA,m.QSpinSyntheticPayloadLifecycle.ROUNDTRIP_STUBBED,m.QSpinSyntheticPayloadLifecycle.PERMISSION_CHECKED,m.QSpinSyntheticPayloadLifecycle.EXECUTOR_DRY_RUN_PASSED,m.QSpinSyntheticPayloadLifecycle.DISCARDED))

def req(**kw):
    base=dict(roundtrip_approved=True,permission_approved=True,executor_approved=True,payload_dry_run_approved=True,dispatch_approved=True,kill_switch_allows=True,rollback_evidence_present=True,commit_gate_approved=True)
    base.update(kw)
    return m.QSpinSyntheticPayloadExecutionRequest("r",env(),**base)

def test_all_kinds_execute_synthetic_only():
    h=m.build_default_qspin_synthetic_payload_execution_harness()
    for kind in m.QSpinSyntheticPayloadKind:
        r=req(); r=m.QSpinSyntheticPayloadExecutionRequest(kind.value,env(kind),True,True,True,True,True,True,True,True)
        out=h.execute(r)
        assert out.decision.executed is True
        assert out.transferred_payload is False and out.wrote_state is False

def test_deterministic_synthetic_payload_id():
    assert env().synthetic_payload_id()==env().synthetic_payload_id()

def test_unsafe_shape_budget_and_missing_approval_block():
    h=m.build_default_qspin_synthetic_payload_execution_harness()
    bad_env=m.QSpinSyntheticPayloadEnvelope(m.QSpinSyntheticPayloadKind.DENSE,"s","t",m.QSpinSyntheticPayloadShape((9999999,9999999)),m.QSpinSyntheticPayloadBudget(1))
    out=h.execute(m.QSpinSyntheticPayloadExecutionRequest("bad",bad_env,True,True,True,True,True,True,True,True))
    assert out.decision.executed is False
    out2=h.execute(req(roundtrip_approved=False))
    assert m.QSpinSyntheticPayloadBlockReason.MISSING_ROUNDTRIP_APPROVAL in out2.decision.block_reasons

def test_no_live_effect_result_enforced():
    h=m.build_default_qspin_synthetic_payload_execution_harness(); out=h.execute(req())
    try:
        m.QSpinSyntheticPayloadExecutionResult(out.request,out.decision,out.trace,out.audit_event,transferred_payload=True).validate()
    except ValueError: pass
    else: raise AssertionError("live effect must be rejected")
