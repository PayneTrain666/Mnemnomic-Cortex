from tests.qspin_prod4_module_loader import load_prod4_modules
m=load_prod4_modules()["qspin_synthetic_qh_shared_slot_sandbox"]

def request(scope=m.QSpinSyntheticSandboxScope.SHARED_SLOT, op=m.QSpinSyntheticSandboxOperation.READ_METADATA, **kw):
    base=dict(interference_check_present=True,commit_gate_review_present=True,permission_metadata_present=True)
    base.update(kw)
    return m.QSpinSyntheticSandboxRequest("rq",scope,op,"target",**base)

def test_sandbox_read_and_synthetic_writes():
    s=m.build_default_qspin_synthetic_qh_shared_slot_sandbox()
    assert s.operate(request()).decision.allowed is True
    assert s.operate(request(op=m.QSpinSyntheticSandboxOperation.WRITE_SANDBOX)).decision.allowed is True
    assert len(s.shared_slot_records)==1
    assert s.operate(request(scope=m.QSpinSyntheticSandboxScope.QH,op=m.QSpinSyntheticSandboxOperation.WRITE_SANDBOX)).decision.allowed is True
    assert len(s.qh_records)==1

def test_real_writes_and_missing_checks_block():
    s=m.build_default_qspin_synthetic_qh_shared_slot_sandbox()
    out=s.operate(request(op=m.QSpinSyntheticSandboxOperation.WRITE_REAL))
    assert m.QSpinSyntheticSandboxBlockReason.REAL_WRITE_REQUESTED in out.decision.block_reasons
    out2=s.operate(request(interference_check_present=False))
    assert m.QSpinSyntheticSandboxBlockReason.INTERFERENCE_CHECK_MISSING in out2.decision.block_reasons

def test_cleanup_idempotent_and_no_real_store_touch():
    s=m.build_default_qspin_synthetic_qh_shared_slot_sandbox()
    s.operate(request(op=m.QSpinSyntheticSandboxOperation.WRITE_SANDBOX))
    c1=s.cleanup(); c2=s.cleanup()
    assert c1.decision.allowed is True and c2.decision.allowed is True
    assert len(s.shared_slot_records)==0
