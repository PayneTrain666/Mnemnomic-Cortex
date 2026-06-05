from mnemonic_cortex.reasoning_depth import DepthRouteStrategySelector, DepthRouteStrategyConfig, ReasoningTaskMode


def test_reason2b_route_strategy_modes_are_bounded():
    selector = DepthRouteStrategySelector(DepthRouteStrategyConfig(max_depths_per_route=4, max_hops=3))
    plan = selector.select(task_mode=ReasoningTaskMode.HYPOTHESIS, uncertainty=0.8)

    assert len(plan.selected_depths) <= 4
    assert all(0 <= depth <= 7 for depth in plan.selected_depths)
    assert 1 <= plan.max_hops <= 3
    assert 6 in plan.selected_depths
    assert plan.to_dict()["safety"]["non_mutating"] is True
