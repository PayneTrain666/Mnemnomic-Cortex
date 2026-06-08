import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.triple_hybrid import EnhancedTripleHybridMemory


def test_triple_hybrid_spatial_ltm_bank_read_write():
    model = EnhancedTripleHybridMemory(
        input_dim=32,
        output_dim=32,
        enable_spatial_ltm=True,
        spatial_slots=64,
        spatial_value_dim=32,
        spatial_key_dim=16,
    )
    x = torch.randn(2, 4, 32)
    fused = model(x, operation="read")
    assert fused.shape == (2, 4, 32)
    assert torch.isfinite(fused).all()
    assert model.spatial_ltm is not None

    model(x, operation="write")
    spatial_read = model.read_bank("spatial", x)
    assert spatial_read.shape == (2, 4, 32)
    assert torch.isfinite(spatial_read).all()


def test_triple_hybrid_spatial_disabled_keeps_three_bank_router():
    model = EnhancedTripleHybridMemory(
        input_dim=16,
        output_dim=16,
        enable_spatial_ltm=False,
    )
    assert model.spatial_ltm is None
    x = torch.randn(1, 3, 16)
    out = model(x, operation="read")
    assert out.shape == (1, 3, 16)


def test_cortex_wires_spatial_ltm_params():
    cortex = EnhancedMnemonicCortex(
        input_dim=32,
        output_dim=32,
        ltm_enable_spatial_ltm=True,
        ltm_spatial_slots=48,
        ltm_spatial_value_dim=32,
        ltm_n_transformer_layers=3,
        ltm_fusion_transformer_layers=4,
    )
    ltm = cortex.long_term_memory
    assert ltm.enable_spatial_ltm is True
    assert ltm.spatial_ltm is not None
    assert cortex.spatial_ltm_extension is not None
    assert cortex.spatial_wm_lattice_mirror is not None
    assert cortex.spatial_wm_lattice_mirror.num_depths == 8
    core = ltm.spatial_ltm.memory_core
    assert core.transformer_layers == 3
    assert core.fusion_transformer_layers == 4
    assert core.decoder_transformer_layers == 3
    x = torch.randn(1, 2, 32)
    out = cortex.long_term_memory(x, operation="read")
    assert out.shape == (1, 2, 32)


def test_cortex_spatial_ltm_extension_auto_wired():
    cortex = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_enable_spatial_ltm=True,
        ltm_spatial_transformer_layers=2,
        ltm_auto_enable_spatial_extension=True,
    )
    assert cortex.spatial_ltm_extension is not None
    x = torch.randn(2, 16)
    out, traces = cortex.run_spatial_ltm_extension(x, operation="process", return_traces=True)
    assert out.shape == (2, 16)
    assert "ltm" in traces
    assert traces.get("wm_lattice_mirror") is not None
    assert cortex.spatial_ltm_wiring.bank_transformer_layers == 2


def test_spatial_inherits_bank_layers_when_spatial_layers_zero():
    cortex = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_enable_spatial_ltm=True,
        ltm_spatial_transformer_layers=0,
        ltm_n_transformer_layers=3,
    )
    core = cortex.long_term_memory.spatial_ltm.memory_core
    assert core.transformer_layers == 3
    assert core.fixed_transformer_layers == 3
    assert core.aux_transformer is not None


def test_spatial_dual_stack_policy_resolves_both_stacks():
    from mnemonic_cortex.ltm import DualTransformerPolicy

    policy = DualTransformerPolicy.resolve(
        bank_layers_cfg=0,
        fixed_layers_cfg=3,
        inherited_bank_layers=3,
    )
    assert policy.bank_layers == 3
    assert policy.fixed_layers == 3
    assert policy.fixed_layers > 0
    assert policy.to_dict()["dual_stack_active"] is True
