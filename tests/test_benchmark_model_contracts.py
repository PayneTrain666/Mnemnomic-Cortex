from benchmark.models import CortexSeqModel, get_model


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
