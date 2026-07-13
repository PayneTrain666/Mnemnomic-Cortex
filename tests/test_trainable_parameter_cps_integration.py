from __future__ import annotations

import torch

from benchmark.models import CortexSeqModel
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.trainable_parameter_cps import (
    CPSBackedEmbedding,
    CPSBackedLinear,
    TrainableParameterCPSConfig,
)


def _tiny_cortex(**kwargs):
    return EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        **kwargs,
    )


def test_cortex_opt_in_exact_commit_and_safety_contract():
    model = _tiny_cortex()
    model.eval()
    x = torch.randn(3, 16)
    expected = model.ctx_proj(x)
    store = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(include=("ctx_proj",))
    )
    proposal = model.stage_trainable_parameter_consolidation()
    assert proposal.replacements == ("ctx_proj",)
    model.commit_trainable_parameter_consolidation(proposal)

    assert isinstance(model.ctx_proj, CPSBackedLinear)
    assert torch.equal(model.ctx_proj(x), expected)
    desc = model.describe_trainable_parameter_cps()
    assert desc["committed"] is True
    assert desc["capacity"]["literal"] is True
    assert desc["safety"] == {
        "qspin_runtime_activation": False,
        "shared_slot_write": False,
        "qh_storage_write": False,
        "wm_commit_execution": False,
    }
    assert list(model.ctx_proj.parameters()) == []
    assert all(parameter.requires_grad for parameter in store.parameters())


def test_cortex_seq_embedding_and_projection_use_canonical_store():
    model = CortexSeqModel(
        vocab_size=13,
        d_model=16,
        cms_enabled=False,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_enable_spatial_ltm=False,
        ltm_auto_wire_spatial=False,
    )
    token_ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 16)
    expected_embedding = model.embedding(token_ids)
    expected_projection = model.proj(hidden)
    store = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(include=("embedding", "proj"))
    )
    proposal = model.stage_trainable_parameter_consolidation()
    store.commit(proposal)

    assert isinstance(model.embedding, CPSBackedEmbedding)
    assert isinstance(model.proj, CPSBackedLinear)
    assert torch.equal(model.embedding(token_ids), expected_embedding)
    assert torch.equal(model.proj(hidden), expected_projection)
    loss = model.embedding(token_ids).sum() + model.proj(hidden).sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in store.parameters())


def test_cortex_checkpoint_rebuilds_bindings_before_tensor_load(tmp_path):
    model = _tiny_cortex()
    store = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(include=("ctx_proj",))
    )
    store.commit(store.stage(model))
    with torch.no_grad():
        for parameter in store.parameters():
            parameter.add_(0.25)
    sample = torch.randn(2, 16)
    expected = model.ctx_proj(sample)
    path = tmp_path / "trainable_cps.pt"
    model.save_checkpoint(str(path))

    loaded = _tiny_cortex()
    loaded.load_checkpoint(str(path), strict=True)
    assert isinstance(loaded.ctx_proj, CPSBackedLinear)
    assert torch.equal(loaded.ctx_proj(sample), expected)
    assert loaded.describe_trainable_parameter_cps()["committed"] is True


def test_trainable_cps_moves_with_model_device():
    model = _tiny_cortex()
    store = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(include=("ctx_proj",))
    )
    store.commit(store.stage(model))
    store.to(dtype=torch.float64)
    assert all(parameter.dtype == torch.float64 for parameter in store.parameters())
    assert model.ctx_proj(torch.randn(2, 16, dtype=torch.float64)).dtype == torch.float64
    store.to(dtype=torch.float32)
    if not torch.cuda.is_available():
        return
    model.to("cuda")
    assert all(parameter.is_cuda for parameter in store.parameters())
    x = torch.randn(2, 16, device="cuda")
    assert model.ctx_proj(x).is_cuda
    model.to("cpu")
    assert all(not parameter.is_cuda for parameter in store.parameters())


def test_default_is_disabled_and_qspin_filters_are_inert():
    model = _tiny_cortex(
        working_memory_fabric="qdt",
        qdt_qspin_guarded_shadow=True,
        qdt_qspin_live_activation=False,
    )
    assert model.describe_trainable_parameter_cps()["enabled"] is False
    before = model.describe_working_memory_fabric()
    store = model.enable_trainable_parameter_cps(
        TrainableParameterCPSConfig(
            include=("ctx_proj",),
            exclude=("*qspin*",),
        )
    )
    store.commit(store.stage(model))
    after = model.describe_working_memory_fabric()
    assert (
        before["qdt_config"]["qspin_guarded_shadow"]
        == after["qdt_config"]["qspin_guarded_shadow"]
    )
    assert after["qdt_config"]["qspin_live_activation"] is False

