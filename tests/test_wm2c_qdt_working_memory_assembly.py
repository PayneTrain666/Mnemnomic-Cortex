import torch

from mnemonic_cortex.working_memory import QDTWorkingMemoryConfig, QDTWorkingMemory, qdt_config_from_hardware_profile


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


def test_qdt_single_gpu_profile_estimate_and_replica_trace():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    estimate = cfg.capacity_estimate(batch_size=1, seq_len=3).to_dict()
    wm = QDTWorkingMemory(cfg)
    x = torch.randn(1, 3, 32)
    y, trace = wm(x, operation="read", context_map_name="quantum_holographic", return_trace=True)

    assert cfg.hardware_profile == "single_gpu_8_12gb"
    assert cfg.num_depths == 8
    assert cfg.triplet_dim == 3
    assert cfg.qspin_guarded_shadow is True
    assert cfg.qspin_live_activation is True
    assert cfg.qspin_live_mode == "experimental_live"
    assert estimate["depth_state_scalars_per_token"] == 8 * 3 * 32
    assert y.shape == x.shape
    q_depth = [item for item in trace["items"] if item["stage"] == "quaternion_depth"][0]
    assert q_depth["metadata"]["payload"]["output_shape"] == [1, 8, 3, 3, 32]


def test_qdt_guarded_qspin_trace_has_no_live_effects():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    wm = QDTWorkingMemory(cfg)
    _y, trace = wm(torch.randn(1, 2, 32), operation="process", return_trace=True)
    qspin = [item for item in trace["items"] if item["stage"] == "qspin_guarded_shadow"][0]["metadata"]["qspin"]

    assert qspin["enabled"] is True
    assert qspin["mode"] == "guarded_experimental_shadow"
    assert qspin["allowed_shadow_only"] is True
    assert qspin["live_routing"] is False
    assert qspin["payload_transfer"] is False
    assert qspin["writes"] is False
    assert qspin["production_activation"] is False


def test_qdt_experimental_live_qspin_applies_routing_and_payload():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    wm = QDTWorkingMemory(cfg)
    _y, trace = wm(torch.randn(1, 4, 32), operation="read", return_trace=True)
    live_items = [item for item in trace["items"] if item["stage"] == "qspin_experimental_live"]
    decisions = [
        item["metadata"]["qspin"]["decision"]
        for item in live_items
        if "qspin" in item["metadata"]
    ]

    assert decisions[-1]["status"] == "allowed_experimental_live"
    assert decisions[-1]["live_routing"] is True
    assert decisions[-1]["payload_transfer"] is True
    assert any(item["message"] == "live_depth_phase_routing_applied" for item in live_items)
    payload_item = [item for item in live_items if item["message"] == "bounded_payload_transferred_to_attention"][0]
    assert payload_item["metadata"]["payload_shape"] == [1, 4, 32]
    assert payload_item["metadata"]["raw_payload_free"] is True


def test_qdt_experimental_live_qspin_kill_switch_blocks_effects():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    cfg.qspin_live_kill_switch_enabled = False
    wm = QDTWorkingMemory(cfg)
    _y, trace = wm(torch.randn(1, 4, 32), operation="read", return_trace=True)
    live_items = [item for item in trace["items"] if item["stage"] == "qspin_experimental_live"]
    decision = [item["metadata"]["qspin"]["decision"] for item in live_items if "qspin" in item["metadata"]][-1]

    assert decision["status"] == "blocked"
    assert "kill_switch_not_enabled" in decision["block_reasons"]
    assert not any(item["message"] == "live_depth_phase_routing_applied" for item in live_items)
    assert not any(item["message"] == "bounded_payload_transferred_to_attention" for item in live_items)


def test_qdt_experimental_live_qspin_missing_evidence_blocks_effects():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    cfg.qspin_source_matrix_complete = False
    cfg.qspin_rollback_evidence_present = False
    wm = QDTWorkingMemory(cfg)
    _y, trace = wm(torch.randn(1, 4, 32), operation="read", return_trace=True)
    live_items = [item for item in trace["items"] if item["stage"] == "qspin_experimental_live"]
    decision = [item["metadata"]["qspin"]["decision"] for item in live_items if "qspin" in item["metadata"]][-1]

    assert decision["status"] == "blocked"
    assert "source_matrix_incomplete" in decision["block_reasons"]
    assert "rollback_evidence_missing" in decision["block_reasons"]


def test_qdt_experimental_live_qspin_write_uses_gated_paths():
    cfg = qdt_config_from_hardware_profile("single_gpu_8_12gb", input_dim=32)
    cfg.num_slots = 8
    cfg.transformer_layers = 1
    cfg.maae_transformer_layers = 1
    cfg.cross_model_attention_layers = 1
    wm = QDTWorkingMemory(cfg)
    _y, trace = wm(torch.randn(1, 3, 32), operation="write", return_trace=True)
    live_gate = [
        item for item in trace["items"]
        if item["stage"] == "qspin_experimental_live" and item["message"] == "live_write_path_gated"
    ][0]

    assert live_gate["metadata"]["qspin_decision"]["shared_slot_write"] is True
    assert live_gate["metadata"]["qspin_decision"]["qh_storage_write"] is True
    assert live_gate["metadata"]["qspin_decision"]["commit_execution"] is True
    assert live_gate["metadata"]["commit_gate"] is True


def test_qdt_compact_profile_keeps_qspin_live_off_by_default():
    cfg = qdt_config_from_hardware_profile("compact", input_dim=32)

    assert cfg.qspin_live_activation is False
    assert cfg.qspin_live_mode == "disabled"
