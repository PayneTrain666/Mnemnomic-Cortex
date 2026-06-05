import torch

from mnemonic_cortex.working_memory import QDTWorkingMemoryConfig, QDTWorkingMemory


def make_wm():
    cfg = QDTWorkingMemoryConfig(
        input_dim=8,
        hidden_dim=16,
        num_depths=4,
        num_slots=4,
        num_heads=2,
        transformer_layers=1,
    )
    return QDTWorkingMemory(cfg)


def test_qdt_working_memory_read_shape_trace_and_finite():
    wm = make_wm()
    x = torch.randn(1, 3, 8)
    y, trace = wm(x, operation="read", context_map_name="literal", return_trace=True)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert trace["operation"] == "read"
    assert trace["confidence"] > 0.0
    stages = [item["stage"] for item in trace["items"]]
    assert "curved_core" in stages
    assert "quaternion_depth" in stages
    assert "intra_depth" in stages
    assert "cross_depth" in stages
    assert "depth_adapters" in stages
    assert "depth_specific_addressing" in stages
    assert "depth_fusion" in stages


def test_qdt_working_memory_process_shape():
    wm = make_wm()
    x = torch.randn(1, 3, 8)
    y = wm(x, operation="process")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_qdt_working_memory_write_uses_shadow_trace():
    wm = make_wm()
    x = torch.randn(1, 3, 8)
    y, trace = wm(x, operation="write", return_trace=True)

    assert y.shape == x.shape
    assert trace["operation"] == "write"
    assert trace["paamax_metadata"]["write_permission_required"] is True
    assert any(item["stage"] == "curved_core" for item in trace["items"])


def test_qdt_working_memory_stability_report():
    wm = make_wm()
    x = torch.randn(1, 3, 8)
    report = wm.stability_report(x)

    assert report["ok"] is True
    assert report["finite"] is True
    assert report["shape_ok"] is True
    assert report["output_shape"] == [1, 3, 8]


def test_qdt_working_memory_rejects_bad_operation():
    wm = make_wm()
    x = torch.randn(1, 3, 8)
    try:
        wm(x, operation="bad")
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_qdt_working_memory_config_rejects_bad_heads():
    try:
        QDTWorkingMemoryConfig(input_dim=10, num_heads=4).validate()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
