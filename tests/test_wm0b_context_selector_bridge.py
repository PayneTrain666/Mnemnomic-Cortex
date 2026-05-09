import torch

from mnemonic_cortex.working_memory.context_map_selector import ContextMapSelector
from mnemonic_cortex.working_memory.context_to_wm_bridge import ContextToWMBridge


def test_selector_uses_requested_map():
    selector = ContextMapSelector(dim=32)
    ctx = torch.randn(2, 4, 32)
    selected, trace = selector.select(ctx, requested_map="procedural")
    assert selected.name == "procedural"
    assert trace.reason == "requested_map"


def test_selector_uses_task_hints():
    selector = ContextMapSelector(dim=32)
    ctx = torch.randn(2, 4, 32)
    selected, trace = selector.select(ctx, task_hints=["cad", "mechanical", "layout"])
    assert selected.name == "spatial_mechanical"


def test_context_to_wm_bridge_mounts_and_traces():
    bridge = ContextToWMBridge(dim=32, num_depths=8)
    ctx = torch.randn(2, 4, 32)
    depth_state = torch.randn(2, 8, 5, 3, 32)
    mounted, trace = bridge(ctx, depth_state, task_hints=["policy", "audit"], paamax_policy_hint="write_guard")
    assert mounted.shape == depth_state.shape
    assert trace.selected_map in {"policy_governance", "conflict_verification"}
    assert trace.stability["finite"] is True
    assert "policy_lane" in trace.paamax_policy_tags or "audit_required" in trace.paamax_policy_tags or "conflict_gate" in trace.paamax_policy_tags
