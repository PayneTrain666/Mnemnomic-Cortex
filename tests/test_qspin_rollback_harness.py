from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
rb = mods["qspin_rollback_harness"]


def all_evidence():
    return tuple(rb.QSpinRollbackEvidenceRecord("ev_" + kind.value, kind) for kind in rb.QSpinRollbackEvidenceKind)


def test_default_rollback_plan_has_required_steps():
    plan = rb.build_default_qspin_rollback_plan_snapshot()
    kinds = {s.required_evidence for s in plan.steps}
    assert kinds == set(rb.QSpinRollbackEvidenceKind)
    assert plan.dry_run_only is True


def test_complete_evidence_passes_dry_run_without_mutation():
    harness = rb.build_default_qspin_rollback_harness()
    result = harness.dry_run(rb.QSpinRollbackDryRunRequest("rb_ok", all_evidence()))
    assert result.passed is True
    assert result.executed_real_rollback is False
    assert result.mutated_state is False


def test_missing_evidence_blocks():
    harness = rb.build_default_qspin_rollback_harness()
    evidence = tuple(e for e in all_evidence() if e.kind is not rb.QSpinRollbackEvidenceKind.AUDIT_LOG_PRESERVED)
    result = harness.dry_run(rb.QSpinRollbackDryRunRequest("rb_missing", evidence))
    assert result.passed is False
    assert rb.QSpinRollbackEvidenceKind.AUDIT_LOG_PRESERVED in result.missing_evidence
