import torch

from mnemonic_cortex import ParameterStorageLoopConfig, ParameterStorageLoopStack
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.reasoning_depth import (
    ParameterLoopAdapter,
    ParameterLoopAdapterConfig,
    parameter_loop_adapter_contract,
)


def test_parameter_loop_reasoning_adapter_reads_without_mutation():
    loop = ParameterStorageLoopStack(
        ParameterStorageLoopConfig(model_dim=16, parameter_slots_per_layer=2, free_hidden_layers=0)
    )
    adapter = ParameterLoopAdapter(
        ParameterLoopAdapterConfig.enabled_default(key_dim=16, max_context_tokens=6),
        parameter_loop=loop,
    )
    before = loop.visible_parameter_slots.detach().clone()
    out, trace = adapter.read_parameters(torch.randn(2, 3, 16), return_trace=True)

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    assert trace["adapter"] == "parameter_loop_adapter"
    assert trace["read_only"] is True
    assert trace["no_memory_store_mutation"] is True
    assert torch.equal(before, loop.visible_parameter_slots.detach())


def test_parameter_loop_reasoning_adapter_disabled_passes_through():
    adapter = ParameterLoopAdapter(ParameterLoopAdapterConfig.disabled(key_dim=16))
    query = torch.randn(2, 3, 16)
    out, trace = adapter.read_parameters(query, return_trace=True)

    assert out.shape == (2, 16)
    assert trace["enabled"] is False
    assert trace["pass_through"] is True


def test_cortex_builds_parameter_loop_reasoning_adapter():
    model = EnhancedMnemonicCortex(
        input_dim=16,
        output_dim=16,
        ltm_hg_slots=16,
        ltm_cgmn_slots=16,
        ltm_curved_slots=8,
        ltm_spatial_slots=8,
        enable_parameter_storage_loop_stack=True,
        parameter_loop_slots_per_layer=2,
        parameter_loop_free_hidden_layers=0,
    )
    adapter = model.build_parameter_loop_reasoning_adapter(max_context_tokens=4)
    out, trace = adapter.read_parameters(torch.randn(1, 2, 16), return_trace=True)

    assert out.shape == (1, 16)
    assert trace["enabled"] is True
    assert parameter_loop_adapter_contract()["read_only"] is True
