import torch
import torch.nn as nn
from .memory_hg import EnhancedHyperGeometricMemory
from .memory_cgmn import EnhancedCGMNMemory
from .memory_curved import EnhancedCurvedMemory
from .topology_manager import TopologyManagerV3

class EnhancedTripleHybridMemory(nn.Module):
    """Wrapper combining HG, CGMN, and Curved memories with selectable fusion:
       - 'weighted' (default): learned static mix
       - 'cross_attn': cross-attention over {HG,CGMN,Curved} per timestep
    Also propagates temperature and energy-efficient mode; exposes consolidation.
    """
    def __init__(self, input_dim: int, output_dim: int, hg_slots: int = 2048, cgmn_slots: int = 1024, curved_slots: int = 512, fusion: str = 'weighted',
                 hg_ann_centroids: int = 256, hg_ann_top: int = 8):
        super().__init__()
        self.hg = EnhancedHyperGeometricMemory(input_dim, mem_slots=hg_slots,
                                                ann_centroids=hg_ann_centroids,
                                                ann_top_centroids=hg_ann_top)
        self.cgmn = EnhancedCGMNMemory(input_dim, mem_slots=cgmn_slots)
        self.curved = EnhancedCurvedMemory(input_dim, mem_slots=curved_slots)
        self.topology_manager = TopologyManagerV3(subsystems=("hg", "cgmn", "curved"))
        self.mix = nn.Parameter(torch.tensor([0.34, 0.33, 0.33]))  # [hg, cgmn, curved]
        self.fusion_mode = fusion
        self.cross_fuser = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))

    def set_temperature(self, t: torch.Tensor):
        self.hg.set_temperature(t)
        self.cgmn.set_temperature(t)
        self.curved.set_temperature(t)

    def enable_energy_efficient_mode(self, enable: bool = True):
        self.hg.enable_energy_efficient_mode(enable)
        self.cgmn.enable_energy_efficient_mode(enable)
        self.curved.enable_energy_efficient_mode(enable)

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1):
        self.hg.consolidate_unused(threshold)
        self.cgmn.consolidate_unused(threshold)
        self.curved.consolidate_unused(threshold)

    @torch.no_grad()
    def evolve_topologies(self, fitness_by_subsystem):
        """Update topology states using subsystem fitness values.
        fitness_by_subsystem keys: 'hg' | 'cgmn' | 'curved'
        """
        out = {}
        for name in ("hg", "cgmn", "curved"):
            if name in fitness_by_subsystem:
                out[name] = self.topology_manager.evolve_topology(
                    fitness=float(fitness_by_subsystem[name]),
                    subsystem=name,
                )
                # Apply a small channel-prior nudge to each merger.
                if name == "hg":
                    self.topology_manager.steer_merger(self.hg.geometry_merger, subsystem="hg")
                elif name == "cgmn":
                    self.topology_manager.steer_merger(self.cgmn.geometry_merger, subsystem="cgmn")
                else:
                    self.topology_manager.steer_merger(self.curved.geometry_merger, subsystem="curved")
        return out

    @torch.no_grad()
    def mutate_curvatures(self, loss_value: float):
        """Apply loss-aware curvature mutation via TopologyManagerV3."""
        self.hg.memory_curvature.data.copy_(
            self.topology_manager.mutate_curvature(
                self.hg.memory_curvature.data, loss_value=loss_value, subsystem="hg"
            )
        )
        self.cgmn.curvature.data.copy_(
            self.topology_manager.mutate_curvature(
                self.cgmn.curvature.data, loss_value=loss_value, subsystem="cgmn"
            )
        )
        self.curved.memory_curvature.data.copy_(
            self.topology_manager.mutate_curvature(
                self.curved.memory_curvature.data,
                loss_value=loss_value,
                subsystem="curved",
            )
        )
        # Mutate SPD factors with the same topology mode, gently.
        self.hg.spd_L.data.copy_(self.topology_manager.mutate_tensor_like(self.hg.spd_L.data, subsystem="hg"))
        self.cgmn.spd_L.data.copy_(self.topology_manager.mutate_tensor_like(self.cgmn.spd_L.data, subsystem="cgmn"))
        self.curved.spd_L.data.copy_(self.topology_manager.mutate_tensor_like(self.curved.spd_L.data, subsystem="curved"))

    def _fuse(self, rhg, rcg, rcv):
        if self.fusion_mode == 'cross_attn':
            B,S,D = rhg.shape
            tokens = torch.stack([rhg, rcg, rcv], dim=2)     # (B,S,3,D)
            tokens = tokens.view(B*S, 3, D)                  # (B*S,3,D)
            fused, _ = self.cross_fuser(tokens, tokens, tokens)  # (B*S,3,D)
            fused = fused.mean(dim=1).view(B,S,D)            # (B,S,D)
            return fused
        else:
            w = torch.softmax(self.mix, dim=0)
            return w[0]*rhg + w[1]*rcg + w[2]*rcv

    def forward(self, x: torch.Tensor, operation: str = 'read', fire_mask=None, recall_boost: float = 0.3):
        if operation == 'write':
            self.hg(x, operation='write', fire_mask=fire_mask, recall_boost=recall_boost)
            self.cgmn(x, operation='write', fire_mask=fire_mask, recall_boost=recall_boost)
            self.curved(x, operation='write', importance=None)
            return x
        rhg = self.hg(x, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)       # (B,S,d)
        rcg = self.cgmn(x, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)     # (B,S,d)
        rcv = self.curved(x, operation='read')   # (B,S,d)
        fused = self._fuse(rhg, rcg, rcv)
        return fused
