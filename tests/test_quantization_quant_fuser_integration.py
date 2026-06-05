import torch

from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.cps import ConsolidatedParamStore, UnifiedParamCfg
from mnemonic_cortex.quant_fuser import QuantAwareCPSFuser
from mnemonic_cortex.quantization import CPSQuantizer, QuantPolicy, dequant_int8_sym, quantize_int8_sym


def test_grouped_int8_quantization_roundtrip_is_stable():
    x = torch.randn(96)
    q, s = quantize_int8_sym(x, group_size=32)
    assert q.shape == x.shape
    assert s.shape == (3,)
    xhat = dequant_int8_sym(q, s)
    assert xhat.shape == x.shape
    assert torch.isfinite(xhat).all()
    mse = torch.mean((x - xhat) ** 2).item()
    assert mse < 0.05


def test_quant_aware_fuser_accepts_qpack_path():
    store = ConsolidatedParamStore(UnifiedParamCfg(d_euclid=24, d_hyp=8, d_spher=8, d_fisher=8, d_phase=4))
    up = store.ensure("core:test")
    quant = CPSQuantizer(QuantPolicy())
    qpack = quant.quantize_entry(up)
    fuser = QuantAwareCPSFuser(
        d_out=24,
        dims={"E": 24, "H": 8, "S": 8, "F": 8, "T": 2, "P": 4},
        quantizer=quant,
    )
    fused, aux = fuser(qpack=qpack)
    assert list(fused.shape) == [24]
    assert torch.isfinite(fused).all()
    assert "weights" in aux
    assert len(aux["weights"]) > 0


def test_cortex_process_path_uses_quantized_fusion_when_advanced_stack_enabled():
    torch.manual_seed(0)
    model = EnhancedMnemonicCortex(input_dim=16, output_dim=16)
    model.enable_advanced_consolidation(cps_domains=["core", "reasoning"])
    x = torch.randn(2, 4, 16)
    ctx = torch.randn(2, 16)
    token_ids = torch.randint(0, 32, (2, 4))
    _ = model(
        x,
        ctx,
        operation="process",
        token_ids=token_ids,
        use_consolidated_memory=True,
    )
    assert isinstance(model.last_cps_aux, dict)
    assert int(model.last_cps_aux.get("quantized_fusion_count", 0)) > 0
