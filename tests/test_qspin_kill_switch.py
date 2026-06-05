from tests.qspin_prod1_module_loader import load_prod1_modules
mods = load_prod1_modules()
ks = mods["qspin_kill_switch"]


def test_enabled_default_allows_shadow_decision():
    switch = ks.build_default_qspin_kill_switch()
    assert switch.state is ks.QSpinKillSwitchState.ENABLED
    assert switch.decision().allowed_for_shadow is True


def test_trip_blocks_shadow_and_audits():
    switch = ks.build_default_qspin_kill_switch()
    result = switch.trip()
    assert result.state is ks.QSpinKillSwitchState.TRIPPED
    assert switch.decision().allowed_for_shadow is False
    assert len(switch.audit_events) == 1


def test_disabled_invalid():
    switch = ks.QSpinKillSwitch(state=ks.QSpinKillSwitchState.DISABLED)
    try:
        switch.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("disabled kill switch must be invalid")


def test_reset_requires_dry_run_approval_and_does_not_activate_runtime():
    switch = ks.build_default_qspin_kill_switch()
    switch.trip()
    try:
        switch.reset(ks.QSpinKillSwitchResetRequest("reset", False))
    except ValueError:
        pass
    else:
        raise AssertionError("reset without approval should fail")
    decision = switch.reset(ks.QSpinKillSwitchResetRequest("reset_ok", True))
    assert decision.reset_allowed is True
    assert decision.runtime_activated is False
