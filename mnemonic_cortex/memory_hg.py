import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import fast_pairwise_l2

class EnhancedHyperGeometricMemory(nn.Module):
    """HyperGeometric-ish memory with fractal addressing and quantum-inspired phase modulation.
    Shapes are consistent; reads return (B,S,input_dim).
    """
    def __init__(self, input_dim: int, manifold_dim: int = 24, mem_slots: int = 1024,
                 quantum_qubits: int = 8, topk: int = 32, fractal_scales: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.D = manifold_dim
        self.M = mem_slots
        self.Q = quantum_qubits
        self.K = min(topk, mem_slots)
        self.SCALES = fractal_scales

        # Memory keys/values in addressing/manifold space
        self.keys   = nn.Parameter(torch.randn(self.M, self.D))
        self.values = nn.Parameter(torch.randn(self.M, self.D * 3))

        # Quantum-ish per-slot phases and gain
        self.quantum_phase = nn.Parameter(torch.randn(self.M, self.Q))
        self.quantum_gain  = nn.Parameter(torch.ones(self.M))

        # Ricci flow operator across the manifold dimension
        self.ricci_flow = nn.Parameter(torch.eye(self.D))

        # Projections
        self.input_projection  = nn.Sequential(nn.Linear(input_dim, self.D * 3), nn.LayerNorm(self.D * 3), nn.GELU())
        self.output_projection = nn.Sequential(nn.Linear(self.D * 3, input_dim), nn.LayerNorm(input_dim), nn.GELU())

        # Learnable weights per fractal scale
        self.fractal_weights = nn.Parameter(torch.ones(self.SCALES))

        # Temperature used to scale distances (can be modulated by ExplosiveRecallScaler)
        self.register_buffer("temperature", torch.tensor(1.0))

    def set_temperature(self, t: torch.Tensor):
        # t: scalar tensor or (B,) broadcastable later; store scalar fallback
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
            self.temperature = t.mean().detach()

    def encode_to_manifold(self, x):
        # x: (B,S,input_dim) -> (B,S,D,3)
        B, S, _ = x.shape
        z = self.input_projection(x)             # (B,S,3D)
        z = z.view(B, S, self.D, 3)              # (B,S,D,3)
        z = torch.einsum('bsdq,dd->bsdq', z, self.ricci_flow)  # Ricci map
        return z

    @torch.no_grad()
    def _fractal_cdist(self, q: torch.Tensor, k: torch.Tensor):
        # q: (B,S,D)  k: (M,D) -> (B,S,M)
        weights = torch.softmax(self.fractal_weights, dim=0)
        dsum = 0.0
        for s, w in enumerate(weights):
            sf = 2 ** s
            d = fast_pairwise_l2(q / sf, k / sf)  # (B,S,M)
            dsum = dsum + w * d
        return dsum

    def forward(self, x: torch.Tensor, operation: str = 'read'):
        # x: (B,S,input_dim)
        B, S, _ = x.shape
        z = self.encode_to_manifold(x)           # (B,S,D,3)

        if operation == 'write':
            # Minimal write: move values slightly toward current content mean
            content = z.mean(dim=(1,2))          # (B,3)
            # No-op writer for stability; extend with learned writer as needed
            return x

        # READ
        q = z.mean(dim=3)                        # (B,S,D)
        dist = self._fractal_cdist(q, self.keys) # (B,S,M)

        # Temperature scaling (lower temp => sharper)
        dist = dist / self.temperature.clamp_min(1e-6)

        dtop, itop = torch.topk(dist, self.K, dim=-1, largest=False)  # (B,S,K)
        w = F.softmax(-dtop, dim=-1)                                  # (B,S,K)

        # Quantum phase tweak
        phases = torch.tanh(self.quantum_phase[itop])                 # (B,S,K,Q)
        phase_gain = torch.tanh(phases).mean(dim=-1)                  # (B,S,K)
        w = F.softmax(-dtop + 0.05 * phase_gain, dim=-1)

        v = self.values[itop]                           # (B,S,K,3D)
        read = torch.sum(w.unsqueeze(-1) * v, dim=2)    # (B,S,3D)
        out = self.output_projection(read)              # (B,S,input_dim)
        return out
