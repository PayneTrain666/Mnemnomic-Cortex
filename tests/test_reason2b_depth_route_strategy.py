from mnemonic_cortex.reasoning_depth import DepthRouteStrategySelector, DepthRouteStrategyConfig, ReasoningTaskMode


def test_reason2b_route_strategy_modes_are_bounded():
    selector = DepthRouteStrategySelector(DepthRouteStrategyConfig(max_depths_per_route=4, max_hops=3))
    plan = selector.select(task_mode=ReasoningTaskMode.HYPOTHESIS, uncertainty=0.8)

    assert len(plan.selected_depths) <= 4
    assert all(0 <= depth <= 7 for depth in plan.selected_depths)
    assert 1 <= plan.max_hops <= 3
    assert 6 in plan.selected_depths
    assert plan.to_dict()["safety"]["non_mutating"] is True


def test_reason2b_route_strategy_selects_canonical_ltm_banks_by_task():
    selector = DepthRouteStrategySelector(DepthRouteStrategyConfig(max_depths_per_route=4, max_hops=3))

    structural = selector.select(task_mode=ReasoningTaskMode.STRUCTURAL_REASONING)
    hypothesis = selector.select(task_mode=ReasoningTaskMode.HYPOTHESIS)
    episodic = selector.select(task_mode=ReasoningTaskMode.DEFAULT, content_hint="project episode chat")
    spatial = selector.select(task_mode=ReasoningTaskMode.DEFAULT, content_hint="spatial map quaternion pose")

    assert structural.preferred_ltm_bank == "curved_associative"
    assert hypothesis.preferred_ltm_bank == "procedural_spcp"
    assert episodic.preferred_ltm_bank == "hg_episodic"
    assert spatial.preferred_ltm_bank == "spatial_topological"
