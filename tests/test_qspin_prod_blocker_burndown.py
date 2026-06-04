from qspin_prod8_module_loader import load_module
m = load_module("qspin_prod_blocker_burndown")

def test_blocker_burndown_preserves_p0_p1():
    report = m.ProductionBlockerBurnDownPlanner().plan(m.build_default_blocker_burndown_plan())
    assert report.p0_blockers
    assert report.p1_blockers
    assert report.status.value == "blocked"
