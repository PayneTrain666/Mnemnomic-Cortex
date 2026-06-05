import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.working_memory import RuntimeExternalMemoryBank


def test_full_stack_wires_runtime_wm_external_banks():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_cps_cms_full_stack(
        vocab_size=64,
        cms_senses=3,
        enable_broker=False,
        enable_advanced=True,
        enable_reasoning_bridge=True,
        enable_qdt_wm_bridge=True,
    )
    qdt = model._resolve_qdt_working_memory()
    assert qdt is not None
    assert isinstance(qdt.dual_fusion.ltm.external_bank, RuntimeExternalMemoryBank)
    assert isinstance(qdt.dual_fusion.mann.external_bank, RuntimeExternalMemoryBank)
    assert isinstance(qdt.dual_fusion.spcp.external_bank, RuntimeExternalMemoryBank)

    x = torch.randn(2, 4, 16)
    ctx = torch.randn(2, 16)
    tok = torch.randint(low=0, high=64, size=(2, 4))
    out = model(
        x,
        ctx,
        operation="process",
        token_ids=tok,
        use_consolidated_memory=True,
        context_features=x,
    )
    assert out.shape == x.shape


def test_advanced_broker_ingests_ltm_merges_from_encode():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_advanced_consolidation()
    info = torch.randn(2, 3, 16)
    context = torch.randn(2, 16)
    _ = model.encode_memory(info, context, mtype="episodic")
    keys = model.advanced_cms.keys()
    assert len(keys) >= 1
    assert any(str(k).startswith("ltm:episodic:") for k in keys)


def test_retrieve_path_with_reasoning_bridge_is_stable():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_reasoning_controller_bridge(enabled=True, allow_shared_mann_ltm_geometry=True)
    cue = torch.randn(2, 4, 16)
    context = torch.randn(2, 16)
    out = model.retrieve_memory(cue, context, strategy="direct")
    assert out.shape == (2, 16)


def test_qdt_bridge_auto_selects_valid_head_count():
    model = EnhancedMnemonicCortex(input_dim=32, output_dim=32)
    model.enable_qdt_working_memory_bridge(num_heads=None)
    qdt = model._resolve_qdt_working_memory()
    assert qdt is not None
