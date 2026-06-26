import tempfile
from pathlib import Path

from mnemonic_cortex.config_loader import (
    build_cortex_from_yaml,
    load_unified_yaml_config,
    load_yaml_config,
)
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_unified_yaml_loader_parses_cortex_global_and_features():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "unified.yaml"
        p.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  input_dim: 32",
                    "  output_dim: 32",
                    "  capacity_profile: compact",
                    "  wm_slots: 9",
                    "  wm_slot_dim: 192",
                    "  fusion: cross_attn",
                    "  hgm_enabled: false",
                    "  ltm_hg_slots: 96",
                    "  ltm_cgmn_slots: 80",
                    "  ltm_curved_slots: 40",
                    "  ltm_spatial_slots: 72",
                    "  max_external_context_tokens: 24",
                    "  max_parameter_tokens: 20",
                    "  enable_parameter_storage_loop_stack: true",
                    "  parameter_loop_slots_per_layer: 3",
                    "  parameter_loop_free_hidden_layers: 1",
                    "  enable_parameter_loop_ltm_context: true",
                    "  enable_parameter_loop_training_writes: false",
                    "  parameter_loop_training_write_scale: 0.5",
                    "  working_memory_fabric: qdt",
                    "  qdt_hardware_profile: single_gpu_8_12gb",
                    "  qdt_num_slots: 6",
                    "  qdt_transformer_layers: 1",
                    "  qdt_qspin_guarded_shadow: true",
                    "features:",
                    "  hgm_enabled: true",
                    "  reasoning_bridge_enabled: false",
                    "stores:",
                    "  SKS:",
                    "    senses: 4",
                    "    conformal_b: 0.03",
                    "    geometries:",
                    "      hyperbolic: 0.4",
                    "      phase: 0.3",
                    "      fisher: 0.3",
                    "ahg:",
                    "  ask_on_uncertain: \"true\"",
                    "distill:",
                    "  enabled: true",
                    "  student_domains: [reasoning]",
                ]
            ),
            encoding="utf-8",
        )
        unified = load_unified_yaml_config(str(p))
    assert unified.cortex.input_dim == 32
    assert unified.cortex.fusion == "cross_attn"
    assert unified.cortex.capacity_profile == "compact"
    assert unified.cortex.ltm_hg_slots == 96
    assert unified.cortex.max_external_context_tokens == 24
    assert unified.cortex.enable_parameter_storage_loop_stack is True
    assert unified.cortex.parameter_loop_slots_per_layer == 3
    assert unified.cortex.parameter_loop_free_hidden_layers == 1
    assert unified.cortex.enable_parameter_loop_ltm_context is True
    assert unified.cortex.enable_parameter_loop_training_writes is False
    assert unified.cortex.parameter_loop_training_write_scale == 0.5
    assert unified.cortex.working_memory_fabric == "qdt"
    assert unified.cortex.qdt_hardware_profile == "single_gpu_8_12gb"
    assert unified.cortex.qdt_num_slots == 6
    assert unified.cortex.qdt_transformer_layers == 1
    assert unified.cortex.qdt_qspin_guarded_shadow is True
    # Features section can promote constructor flag.
    assert unified.cortex.hgm_enabled is True
    assert unified.features.hgm_enabled is True
    assert unified.global_config.ahg.ask_on_uncertain is True
    assert unified.global_config.distill.enabled is True
    assert "SKS" in unified.global_config.stores


def test_load_yaml_config_remains_backward_compatible():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "legacy.yaml"
        p.write_text(
            "\n".join(
                [
                    "stores: {}",
                    "ahg:",
                    "  ask_on_uncertain: \"false\"",
                    "distill:",
                    "  enabled: \"true\"",
                    "  student_domains: [reasoning, science]",
                ]
            ),
            encoding="utf-8",
        )
        cfg = load_yaml_config(str(p))
    assert cfg.ahg.ask_on_uncertain is False
    assert cfg.distill.enabled is True
    assert tuple(cfg.distill.student_domains) == ("reasoning", "science")


def test_build_cortex_from_yaml_constructs_model_from_typed_entrypoint():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "build.yaml"
        p.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  input_dim: 16",
                    "  output_dim: 16",
                    "  wm_slots: 5",
                    "  ltm_hg_slots: 40",
                    "  ltm_cgmn_slots: 36",
                    "  ltm_curved_slots: 20",
                    "  enable_parameter_storage_loop_stack: true",
                    "  parameter_loop_slots_per_layer: 2",
                    "  parameter_loop_free_hidden_layers: 1",
                    "  enable_parameter_loop_ltm_context: true",
                    "  working_memory_fabric: qdt",
                    "  qdt_hardware_profile: single_gpu_8_12gb",
                    "  qdt_num_slots: 4",
                    "  qdt_transformer_layers: 1",
                    "  hgm_enabled: true",
                    "stores: {}",
                    "ahg: {}",
                    "distill: {}",
                ]
            ),
            encoding="utf-8",
        )
        model, unified = build_cortex_from_yaml(str(p))
    assert model.input_dim == 16
    assert model.output_dim == 16
    assert model.long_term_memory.hg.M == 40
    assert model.long_term_memory.cgmn.M == 36
    assert model.long_term_memory.curved.M == 20
    assert model.parameter_storage_loop_stack is not None
    assert model.describe_parameter_storage_loop()["enabled"] is True
    fabric = model.describe_working_memory_fabric()
    assert fabric["fabric"] == "qdt"
    assert fabric["qdt_config"]["num_depths"] == 8
    assert fabric["qdt_config"]["num_slots"] == 4
    assert fabric["qdt_config"]["qspin_guarded_shadow"] is True
    assert fabric["migration_trace"]["live_ltm_adapter_attached"] is True
    assert unified.cortex.hgm_enabled is True
    assert model.hgm_enabled is True


def test_configure_from_yaml_applies_features_on_existing_model():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "configure.yaml"
        p.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  input_dim: 16",
                    "  output_dim: 16",
                    "features:",
                    "  hgm_enabled: true",
                    "stores: {}",
                    "ahg: {}",
                    "distill:",
                    "  enabled: true",
                    "  student_domains: [reasoning]",
                ]
            ),
            encoding="utf-8",
        )
        model, _ = build_cortex_from_yaml(str(p))
        model.enable_hypergraph_manifold_bridge(enabled=False)
        unified = model.configure_from_yaml(str(p))
    assert unified.features.hgm_enabled is True
    assert model.hgm_enabled is True
    assert model.distillation_config.enabled is True


def test_configure_from_yaml_can_bootstrap_broker_in_one_call():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "broker.yaml"
        p.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  input_dim: 16",
                    "  output_dim: 16",
                    "stores:",
                    "  SKS:",
                    "    senses: 3",
                    "    conformal_b: 0.02",
                    "    geometries:",
                    "      hyperbolic: 0.4",
                    "      phase: 0.3",
                    "      fisher: 0.3",
                    "ahg: {}",
                    "distill: {}",
                ]
            ),
            encoding="utf-8",
        )
        model, _ = build_cortex_from_yaml(str(p))
        model.consolidation_broker = None
        model.configure_from_yaml(str(p), broker_vocab_size=64)
    assert model.consolidation_broker is not None


def test_cortex_from_yaml_classmethod_builds_and_configures():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "from_yaml.yaml"
        p.write_text(
            "\n".join(
                [
                    "cortex:",
                    "  input_dim: 16",
                    "  output_dim: 16",
                    "  hgm_enabled: true",
                    "stores: {}",
                    "ahg: {}",
                    "distill: {}",
                ]
            ),
            encoding="utf-8",
        )
        model, unified = EnhancedMnemonicCortex.from_yaml(str(p))
    assert isinstance(model, EnhancedMnemonicCortex)
    assert model.input_dim == 16
    assert model.output_dim == 16
    assert model.hgm_enabled is True
    assert unified.cortex.hgm_enabled is True
