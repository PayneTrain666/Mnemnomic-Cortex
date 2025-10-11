import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_hg import EnhancedHyperGeometricMemory
from .memory_cgmn import EnhancedCGMNMemory
from .memory_curved import EnhancedCurvedMemory

class EnhancedTripleHybridMemory(nn.Module):
    """Wrapper combining HG, CGMN, and Curved memories with simple mixing.
    """
    def __init__(self, input_dim: int, output_dim: int, hg_slots: int = 2048, cgmn_slots: int = 1024, curved_slots: int = 512):
        super().__init__()
        self.hg = EnhancedHyperGeometricMemory(input_dim, mem_slots=hg_slots)
        self.cgmn = EnhancedCGMNMemory(input_dim, mem_slots=cgmn_slots)
        self.curved = EnhancedCurvedMemory(input_dim, mem_slots=curved_slots)
        self.mix = nn.Parameter(torch.tensor([0.34, 0.33, 0.33]))  # [hg, cgmn, curved]

        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))

    def set_temperature(self, t: torch.Tensor):
        # propagate temperature into sub-memories
        self.hg.set_temperature(t)
        self.cgmn.set_temperature(t)
        self.curved.set_temperature(t)

    def forward(self, x: torch.Tensor, operation: str = 'read'):
        if operation == 'write':
            self.hg(x, operation='write')
            self.cgmn(x, operation='write')
            self.curved(x, operation='write')
            return x

        rhg = self.hg(x, operation='read')       # (B,S,d)
        rcg = self.cgmn(x, operation='read')     # (B,S,d)
        rcv = self.curved(x, operation='read')   # (B,S,d)

        w = torch.softmax(self.mix, dim=0)
        fused = w[0]*rhg + w[1]*rcg + w[2]*rcv   # (B,S,d)
        return fused
