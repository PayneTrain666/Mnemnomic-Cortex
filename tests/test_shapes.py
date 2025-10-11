import torch
from mnemonic_cortex.cortex import EnhancedMnemonicCortex
from mnemonic_cortex.memory_curved import EnhancedCurvedMemory
from mnemonic_cortex.memory_hg import EnhancedHyperGeometricMemory
from mnemonic_cortex.memory_cgmn import EnhancedCGMNMemory

def test_cortex_shapes():
    B,S,d = 4,6,64
    x = torch.randn(B,S,d)
    ctx = torch.randn(B,d)
    model = EnhancedMnemonicCortex(input_dim=d, output_dim=d)
    proc = model(x, ctx, operation='process')
    assert proc.shape == (B,S,d)
    ret = model(x, ctx, operation='retrieve')
    assert ret.shape == (B,d)

def _check_memory(mem_cls):
    B,S,d = 3,5,32
    x = torch.randn(B,S,d)
    mem = mem_cls(d)
    out = mem(x, operation='read')
    assert out.shape == (B,S,d)

def test_curved_memory():
    _check_memory(EnhancedCurvedMemory)

def test_hg_memory():
    _check_memory(EnhancedHyperGeometricMemory)

def test_cgmn_memory():
    _check_memory(EnhancedCGMNMemory)
