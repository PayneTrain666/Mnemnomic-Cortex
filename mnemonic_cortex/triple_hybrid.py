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
        nheads = self._pick_num_heads(input_dim)
        self.hg = EnhancedHyperGeometricMemory(input_dim, mem_slots=hg_slots,
                                                ann_centroids=hg_ann_centroids,
                                                ann_top_centroids=hg_ann_top)
        self.cgmn = EnhancedCGMNMemory(input_dim, mem_slots=cgmn_slots)
        self.curved = EnhancedCurvedMemory(input_dim, mem_slots=curved_slots)
        self.topology_manager = TopologyManagerV3(subsystems=("hg", "cgmn", "curved"))
        self.mix = nn.Parameter(torch.tensor([0.34, 0.33, 0.33]))  # [hg, cgmn, curved]
        # Geometry-aware router that consumes pooled input + 12 memory diagnostics.
        self.router = nn.Sequential(
            nn.Linear(input_dim + 12, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )
        self.fusion_mode = fusion
        self.cross_fuser = nn.MultiheadAttention(input_dim, num_heads=nheads, batch_first=True)
        # Explicit inter-memory exchange paths across HG/CGMN/Spatial memories.
        self.hg_to_cg_attn = nn.MultiheadAttention(input_dim, num_heads=nheads, batch_first=True)
        self.cg_to_spatial_attn = nn.MultiheadAttention(input_dim, num_heads=nheads, batch_first=True)
        self.spatial_to_hg_attn = nn.MultiheadAttention(input_dim, num_heads=nheads, batch_first=True)
        self.inter_norm_hg = nn.LayerNorm(input_dim)
        self.inter_norm_cg = nn.LayerNorm(input_dim)
        self.inter_norm_spatial = nn.LayerNorm(input_dim)
        self.global_inter_attn = nn.MultiheadAttention(input_dim, num_heads=nheads, batch_first=True)
        self.inter_exchange_gate = nn.Parameter(torch.tensor(0.20))
        self.last_inter_memory_stats = {}
        self.last_router_weights = None
        self.last_router_stats = {}
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nheads,
                dim_feedforward=max(128, input_dim * 2),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )
        self.refiner_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim))

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

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
        self.hg.memory_curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.hg.memory_curvature, loss_value=loss_value, subsystem="hg"
            )
        )
        self.cgmn.curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.cgmn.curvature, loss_value=loss_value, subsystem="cgmn"
            )
        )
        self.curved.memory_curvature.copy_(
            self.topology_manager.mutate_curvature(
                self.curved.memory_curvature,
                loss_value=loss_value,
                subsystem="curved",
            )
        )
        # Mutate SPD factors with the same topology mode, gently.
        self.hg.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.hg.spd_L, subsystem="hg"))
        self.cgmn.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.cgmn.spd_L, subsystem="cgmn"))
        self.curved.spd_L.copy_(self.topology_manager.mutate_tensor_like(self.curved.spd_L, subsystem="curved"))

    @torch.no_grad()
    def step_topology(self, loss_value: float):
        """
        Per-step topology hook for memory-local geometry controllers.
        """
        if hasattr(self.hg, "step_topology"):
            self.hg.step_topology(loss_value)
        if hasattr(self.cgmn, "step_topology"):
            self.cgmn.step_topology(loss_value)
        if hasattr(self.curved, "step_topology"):
            self.curved.step_topology(loss_value)

    def _fuse(self, rhg, rcg, rcv, routing_weights=None):
        if self.fusion_mode == 'cross_attn':
            B,S,D = rhg.shape
            tokens = torch.stack([rhg, rcg, rcv], dim=2)     # (B,S,3,D)
            tokens = tokens.view(B*S, 3, D)                  # (B*S,3,D)
            fused, _ = self.cross_fuser(tokens, tokens, tokens)  # (B*S,3,D)
            fused = fused.mean(dim=1).view(B,S,D)            # (B,S,D)
            refined = self.refiner(fused)
            return self.refiner_norm(fused + refined)
        else:
            if routing_weights is None:
                w = torch.softmax(self.mix, dim=0)
                fused = w[0] * rhg + w[1] * rcg + w[2] * rcv
            else:
                w = routing_weights.unsqueeze(1).unsqueeze(-1)  # (B,1,3,1)
                stacked = torch.stack([rhg, rcg, rcv], dim=2)   # (B,S,3,D)
                fused = (w * stacked).sum(dim=2)
            refined = self.refiner(fused)
            return self.refiner_norm(fused + refined)

    @staticmethod
    def _grab_router_features(mem, bsz: int, device, dtype):
        f = getattr(mem, "last_router_features", None)
        if not isinstance(f, dict):
            z = torch.zeros(bsz, device=device, dtype=dtype)
            return z, z, z, z
        vals = []
        for key in ("omega_mean", "curv_mean", "dist_mean", "entropy"):
            v = f.get(key, None)
            if isinstance(v, torch.Tensor):
                t = v.detach().to(device=device, dtype=dtype).reshape(-1)
                if t.numel() == 1:
                    t = t.expand(bsz)
                elif t.numel() != bsz:
                    t = t.mean().expand(bsz)
            else:
                t = torch.zeros(bsz, device=device, dtype=dtype)
            vals.append(t)
        return tuple(vals)

    def _inter_memory_exchange(self, rhg, rcg, rcv):
        """
        Cross-memory attention cycle:
          HG -> CGMN, CGMN -> Spatial, Spatial -> HG, then global tri-memory pass.
        """
        gate = torch.sigmoid(self.inter_exchange_gate)

        cg_from_hg, w_hg_to_cg = self.hg_to_cg_attn(rcg, rhg, rhg, need_weights=True)
        sp_from_cg, w_cg_to_sp = self.cg_to_spatial_attn(rcv, rcg, rcg, need_weights=True)
        hg_from_sp, w_sp_to_hg = self.spatial_to_hg_attn(rhg, rcv, rcv, need_weights=True)

        rhg_e = self.inter_norm_hg(rhg + gate * hg_from_sp)
        rcg_e = self.inter_norm_cg(rcg + gate * cg_from_hg)
        rcv_e = self.inter_norm_spatial(rcv + gate * sp_from_cg)

        bsz, seq, dim = rhg.shape
        tri = torch.stack([rhg_e, rcg_e, rcv_e], dim=2).reshape(bsz * seq, 3, dim)
        tri_global, w_global = self.global_inter_attn(tri, tri, tri, need_weights=True)
        tri = tri + gate * tri_global
        tri = tri.reshape(bsz, seq, 3, dim)

        rhg_o = tri[:, :, 0, :]
        rcg_o = tri[:, :, 1, :]
        rcv_o = tri[:, :, 2, :]
        self.last_inter_memory_stats = {
            "gate": float(gate.detach().item()),
            "hg_to_cg_mean": float(w_hg_to_cg.detach().mean().item()),
            "cg_to_spatial_mean": float(w_cg_to_sp.detach().mean().item()),
            "spatial_to_hg_mean": float(w_sp_to_hg.detach().mean().item()),
            "global_mean": float(w_global.detach().mean().item()),
        }
        return rhg_o, rcg_o, rcv_o

    def forward(self, x: torch.Tensor, operation: str = 'read', fire_mask=None, recall_boost: float = 0.3):
        if operation == 'write':
            self.hg(x, operation='write', fire_mask=fire_mask, recall_boost=recall_boost)
            self.cgmn(x, operation='write', fire_mask=fire_mask, recall_boost=recall_boost)
            self.curved(x, operation='write', importance=None)
            return x
        rhg = self.hg(x, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)       # (B,S,d)
        rcg = self.cgmn(x, operation='read', fire_mask=fire_mask, recall_boost=recall_boost)     # (B,S,d)
        rcv = self.curved(x, operation='read')   # (B,S,d)
        bsz = x.size(0)
        device = x.device
        dtype = x.dtype
        hg_om, hg_cv, hg_dm, hg_en = self._grab_router_features(self.hg, bsz, device, dtype)
        cg_om, cg_cv, cg_dm, cg_en = self._grab_router_features(self.cgmn, bsz, device, dtype)
        cv_om, cv_cv, cv_dm, cv_en = self._grab_router_features(self.curved, bsz, device, dtype)
        router_feats = torch.stack(
            [
                hg_om, hg_cv, hg_dm, hg_en,
                cg_om, cg_cv, cg_dm, cg_en,
                cv_om, cv_cv, cv_dm, cv_en,
            ],
            dim=-1,
        )  # (B,12)
        router_in = torch.cat([x.mean(dim=1), router_feats], dim=-1)  # (B,input_dim+12)
        routing_weights = self.router(router_in)  # (B,3)
        self.last_router_weights = routing_weights.detach()
        self.last_router_stats = {
            "hg": float(routing_weights[:, 0].detach().mean().item()),
            "cgmn": float(routing_weights[:, 1].detach().mean().item()),
            "curved": float(routing_weights[:, 2].detach().mean().item()),
        }
        rhg, rcg, rcv = self._inter_memory_exchange(rhg, rcg, rcv)
        fused = self._fuse(rhg, rcg, rcv, routing_weights=routing_weights)
        return fused
