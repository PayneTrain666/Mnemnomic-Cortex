import torch

from benchmark.models import CortexSeqModel, get_model
from mnemonic_cortex.cortex import EnhancedMnemonicCortex


def test_get_model_respects_lstm_kwargs():
    model = get_model("lstm", vocab_size=64, d_model=96, num_layers=1)
    assert model.encoder.hidden_size == 96
    assert model.encoder.num_layers == 1


def test_get_model_respects_transformer_kwargs():
    model = get_model("transformer", vocab_size=64, d_model=96, nhead=8, num_layers=1)
    assert model.embedding.embedding_dim == 96


def test_cortex_seq_model_topology_step_is_callable():
    model = CortexSeqModel(vocab_size=128, d_model=64, cms_enabled=False)
    model.topology_step(0.25)


def test_cortex_cps_fusion_works_without_cms_modules():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    sensory = torch.randn(2, 3, 16)
    context = torch.randn(2, 16)
    token_ids = torch.randint(low=0, high=32, size=(2, 3))
    out = model(
        sensory,
        context,
        operation="process",
        token_ids=token_ids,
        use_consolidated_memory=True,
    )
    assert out.shape == sensory.shape
    assert isinstance(model.last_cps_aux, dict)
    assert "agree_loss" in model.last_cps_aux
    assert model.last_cms_aux is None


def test_cortex_seq_model_cms_enabled_populates_cms_aux():
    torch.manual_seed(0)
    model = CortexSeqModel(vocab_size=64, d_model=32, cms_enabled=True)
    src = torch.randint(low=0, high=64, size=(2, 5))
    _ = model(src, return_aux_losses=False)
    assert model.cortex.last_cms_aux is not None
