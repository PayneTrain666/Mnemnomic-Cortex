from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
ex = mods["qspin_active_dry_run_executor"]
rt = mods["qspin_payload_roundtrip_stub"]
perm = mods["qspin_permission_dry_run"]

def assert_raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

def test_malformed_executor_request_rejected():
    assert_raises(lambda: ex.QSpinActiveDryRunExecutionRequest(
        "", "", ex.QSpinActiveDryRunFeatureGate(), False, False, False, False, False, False, False, False
    ).validate())

def test_feature_gate_rejects_production_activation():
    assert_raises(lambda: ex.QSpinActiveDryRunFeatureGate(True, True).validate())

def test_unsafe_roundtrip_request_blocks():
    result = rt.QSpinPayloadCodecRoundtripStub().roundtrip(rt.QSpinPayloadRoundtripRequest(
        "bad", rt.QSpinPayloadCodecStubKind.QH, "src", "dst", (2,), declared_bytes=8, raw_payload={"secret": "no"}
    ))
    assert result.decision.approved is False

def test_write_permission_request_blocks():
    result = perm.QSpinQHSharedSlotPermissionDryRun().check(perm.QSpinPermissionDryRunRequest(
        "write", perm.QSpinPermissionScope.EXTERNAL_MEMORY, perm.QSpinPermissionOperation.WRITE, "ltm",
        interference_check_present=True, commit_gate_review_present=True, external_memory_permission_metadata_present=True,
    ))
    assert result.decision.approved is False
