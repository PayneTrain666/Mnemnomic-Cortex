from tests.qspin_prod3_module_loader import load_prod3_modules
mods = load_prod3_modules()
perm = mods["qspin_permission_dry_run"]

def test_permission_read_simulations_and_write_rejections():
    checker = perm.QSpinQHSharedSlotPermissionDryRun()
    req = perm.QSpinPermissionDryRunRequest(
        "ss_read", perm.QSpinPermissionScope.SHARED_SLOT, perm.QSpinPermissionOperation.READ_SIMULATION, "slot",
        interference_check_present=True, commit_gate_review_present=True, shared_slot_permission_metadata_present=True,
    )
    result = checker.check(req)
    assert result.decision.approved is True
    assert result.performed_real_read is False
    assert result.performed_write is False

    wr = perm.QSpinPermissionDryRunRequest(
        "ss_write", perm.QSpinPermissionScope.SHARED_SLOT, perm.QSpinPermissionOperation.WRITE, "slot",
        interference_check_present=True, commit_gate_review_present=True, shared_slot_permission_metadata_present=True,
    )
    blocked = checker.check(wr)
    assert blocked.decision.approved is False
    assert perm.QSpinPermissionDryRunBlockReason.WRITE_REJECTED in blocked.decision.block_reasons

def test_qh_and_external_metadata_requirements():
    checker = perm.QSpinQHSharedSlotPermissionDryRun()
    qh = perm.QSpinPermissionDryRunRequest(
        "qh", perm.QSpinPermissionScope.QH, perm.QSpinPermissionOperation.READ_SIMULATION, "qh",
        interference_check_present=True, commit_gate_review_present=True, qh_permission_metadata_present=True,
    )
    assert checker.check(qh).decision.approved is True
    ext = perm.QSpinPermissionDryRunRequest(
        "ext", perm.QSpinPermissionScope.EXTERNAL_MEMORY, perm.QSpinPermissionOperation.READ_SIMULATION, "ltm",
        interference_check_present=True, commit_gate_review_present=True, external_memory_permission_metadata_present=True,
    )
    assert checker.check(ext).decision.approved is True

def test_permission_blocks_missing_checks_and_raw_payload():
    checker = perm.QSpinQHSharedSlotPermissionDryRun()
    req = perm.QSpinPermissionDryRunRequest(
        "bad", perm.QSpinPermissionScope.QH, perm.QSpinPermissionOperation.READ_SIMULATION, "qh",
        interference_check_present=False, commit_gate_review_present=False, raw_payload={"x": 1},
    )
    result = checker.check(req)
    reasons = set(result.decision.block_reasons)
    assert perm.QSpinPermissionDryRunBlockReason.INTERFERENCE_CHECK_MISSING in reasons
    assert perm.QSpinPermissionDryRunBlockReason.COMMIT_GATE_REVIEW_MISSING in reasons
    assert perm.QSpinPermissionDryRunBlockReason.RAW_PAYLOAD_PRESENT in reasons
