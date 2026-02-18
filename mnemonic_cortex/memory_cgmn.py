import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .utils import fast_pairwise_l2
from .geometry_merger import GeometryMergerV3
from .geometry_utils import HolonomyProbe, qexp, qmul, qnormalize
from .holo_head import HoloHead
from .lightbulb_controller import LightbulbController
from .memory.conformal import ConformalMLP, warp_knn_with_stats
from geometry.blend import GeometryBlender
from geometry.metric_heads import GeometryMetric
from topology.manager_v3 import DynamicTopologyManagerV2

class EnhancedCGMNMemory(nn.Module):
    """Curved Geometric Memory Network (CGMN) with lightbulb-aware temperature & plasticity."""
    def __init__(self, input_dim: int, manifold_dim: int = 16, mem_slots: int = 512,
                 slot_dim: int = 256, topk: int = 32, use_per_sample_temp: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.D, self.M, self.H = manifold_dim, mem_slots, slot_dim
        self.K_base = min(topk, mem_slots)
        self.use_per_sample_temp = use_per_sample_temp
        self.temp_variance_threshold = 0.3  # fallback to scalar if variance > this

        self.manifold_projection = nn.Sequential(
            nn.Linear(input_dim, self.D * 3),
            nn.LayerNorm(self.D * 3),
            nn.GELU()
        )
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.positional_encoding = nn.Parameter(torch.randn(self.M, self.D, 3))
        self.curvature = nn.Parameter(torch.randn(self.M, self.D))
        self.curv_alpha = nn.Parameter(torch.tensor(0.1))
        self.curvature_scalar = nn.Parameter(self.curvature.detach().mean(dim=-1))
        self.conformal_b = 0.1
        self.conformal_mlp = ConformalMLP(self.D)
        self.last_router_features = None
        self.geom_blender = GeometryBlender(num_slots=self.M)
        self.topology_v3 = DynamicTopologyManagerV2(bank="cgmn")
        self.topology = self.topology_v3
        self.last_geom_weights = None
        self.metric_dim = 64
        self.geom_metric = GeometryMetric(d_query=self.D, d_key=self.metric_dim, d_metric=self.metric_dim)
        self.metric_keys = nn.Parameter(torch.randn(self.M, self.metric_dim) * 0.02)
        self.geom_gamma = 0.30
        self.q_memory_slots = nn.Parameter(qnormalize(torch.randn(self.M, 4)))
        self.spin_conn = nn.Sequential(
            nn.Linear(self.D * 3, 64), nn.SiLU(), nn.Linear(64, 3)
        )
        self.geometry_merger = GeometryMergerV3(
            q_dim=self.D, m_dim=self.D, spd_rank=4, use_heat_kernel=True
        )
        self.spd_L = nn.Parameter(torch.zeros(self.M, self.D, 4))
        self.geometry_merger.spd_L = self.spd_L
        self.qc_dim = 16
        self.mem_complex = nn.Parameter(
            (
                torch.randn(self.M, self.qc_dim)
                + 1j * torch.randn(self.M, self.qc_dim)
            ).to(torch.complex64)
        )
        self.qc_head = nn.Sequential(
            nn.Linear(self.D, 32), nn.SiLU(), nn.Linear(32, 2 * self.qc_dim)
        )
        self.holo = HoloHead(
            dim=self.D,
            num_slots=self.M,
            temperature=1.0,
            phase_noise_std=0.0,
            lightbulb_thresh=0.92,
            explosive_temp=0.6,
            explosive_alpha=0.6,
        )
        self.qhm_alpha_head = nn.Sequential(
            nn.Linear(self.D, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.lb_ctrl = LightbulbController(
            z_thresh_on=2.0,
            z_thresh_off=1.2,
            cooldown_steps=6,
            max_stage2_steps=2,
            budget_per_100=6,
            prefocus_temp_mult=0.87,
            explosive_temp_mult=0.55,
            prefocus_alpha_boost=0.12,
            explosive_alpha_boost=0.42,
        )
        self.qhm_enabled = True
        self.qhm_alpha_override = None

        # Simple ODE dynamics in manifold space
        self.ode_dynamics = nn.Sequential(nn.Linear(self.D * 3, 128), nn.Tanh(), nn.Linear(128, self.D * 3))
        self.ode_steps = 2
        self.ode_dt = 0.5

        self.output_projection = nn.Sequential(nn.Linear(self.H, input_dim), nn.LayerNorm(input_dim), nn.GELU())

        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M

        # internal lightbulb tracker
        self.register_buffer('lb_top1_avg', torch.tensor(1.0))
        self.lb_momentum = 0.99
        self.lb_drop_ratio = 0.7
        self.hol_probe = HolonomyProbe(
            lambda m: self.spin_conn(m.reshape(m.size(0), -1))
        )

    # ---------------- Controls ----------------
    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        elif self.use_per_sample_temp:
            # Per-sample temperature with variance check
            if t.numel() > 1:
                variance = t.var()
                if variance > self.temp_variance_threshold:
                    # Too unstable, fall back to mean
                    self.temperature = t.mean().detach()
                else:
                    # Keep per-sample temps
                    self.temperature = t.detach()
            else:
                self.temperature = t.detach()
        else:
            # Default: collapse to scalar
            self.temperature = t.mean().detach()

    def set_active_fraction(self, frac: float):
        frac = float(max(0.1, min(1.0, frac)))
        self.active_slots = max(8, int(self.M * frac))

    def enable_energy_efficient_mode(self, enable: bool = True):
        self.set_active_fraction(0.5 if enable else 1.0)

    @torch.no_grad()
    def step_topology(self, loss_value: float):
        fit = self.topology.update_fitness(float(loss_value))
        self.topology.pick_mode()
        curv_scalar = self.topology.curvature_scalar(self.curvature)
        _, wext = self.topology.mode_weights()
        curv_new = self.topology.mutate_curvature(curv_scalar, wext)
        if self.curvature.ndim == 2:
            self.curvature.copy_(curv_new.unsqueeze(-1).expand_as(self.curvature))
        else:
            self.curvature.copy_(curv_new)
        self.curvature_scalar.copy_(curv_new)
        pri = self.topology.mode_priors().to(self.metric_keys.device)
        self.geom_blender.set_mode_priors(pri, mix=0.05)
        telem = {
            "entropy": float(self.last_router_features.get("entropy", torch.tensor(0.5)).mean().item())
            if isinstance(self.last_router_features, dict)
            else 0.5,
            "dist_mean": float(self.last_router_features.get("dist_mean", torch.tensor(1.0)).mean().item())
            if isinstance(self.last_router_features, dict)
            else 1.0,
        }
        self.conformal_b = self.topology.schedule_conformal_b(self.conformal_b, telemetry=telem)
        return fit

    # ---------- DDP sync -------------
    @torch.no_grad()
    def _sync_buffers_ddp(self):
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        for t in [self.memory_slots, self.usage_counts]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1, ema: float = 0.9):
        if self.usage_counts.max() <= 0:
            return
        usage_ratio = self.usage_counts / (self.usage_counts.max() + 1e-6)
        mask = usage_ratio < threshold
        if mask.any():
            mean_slot = self.memory_slots[~mask].mean(dim=0, keepdim=True) if (~mask).any() else self.memory_slots.mean(dim=0, keepdim=True)
            self.memory_slots[mask].copy_(ema * self.memory_slots[mask] + (1 - ema) * mean_slot)
            self.usage_counts[mask] = 0.0

    def get_metrics(self):
        """Return dict of diagnostic metrics."""
        M = self.active_slots
        m = {
            'cgmn_temp': self.temperature.mean().item() if self.temperature.numel() > 1 else self.temperature.item(),
            'cgmn_topk_base': self.K_base,
            'cgmn_active_slots': M,
            'cgmn_usage_mean': self.usage_counts[:M].mean().item(),
            'cgmn_usage_max': self.usage_counts[:M].max().item(),
            'cgmn_lb_top1_avg': self.lb_top1_avg.item(),
            'cgmn_conformal_b': float(self.conformal_b),
            'cgmn_topology_fitness_ema': float(self.topology_v3.fitness_ema or 0.0),
        }
        if hasattr(self.topology_v3, "mode"):
            mode_map = {"hyperbolic": 0.0, "spherical": 1.0, "euclidean": 2.0, "fractal": 3.0}
            m["cgmn_topology_mode"] = mode_map.get(str(self.topology_v3.mode), -1.0)
        if isinstance(self.last_geom_weights, dict):
            for k in ("hyperbolic", "spherical", "euclidean", "fractal", "torus", "cp"):
                if k in self.last_geom_weights:
                    m[f"cgmn_geom_w_{k}"] = float(self.last_geom_weights[k])
        return m

    # ---------------- Core ----------------
    def manifold_ode_step(self, x):
        B,S,D,_ = x.shape
        x_flat = x.view(B,S,-1)
        dx = self.ode_dynamics(x_flat).view_as(x)
        return x + self.ode_dt * dx

    def _evolve(self, man):
        x = man
        for _ in range(self.ode_steps):
            x = self.manifold_ode_step(x)
        return x

    def _query_quat_from_manifold(self, manifold_x):
        # manifold_x: (..., D, 3) -> (..., 4)
        v = manifold_x.mean(dim=-2)
        ones = torch.ones_like(v[..., :1])
        return qnormalize(torch.cat([ones, 0.1 * v], dim=-1))

    def _parallel_transport_spin(self, q, manifold_x, dt=1.0):
        lead_shape = manifold_x.shape[:-2]
        feat = manifold_x.reshape(-1, self.D * 3)
        omega = self.spin_conn(feat).view(*lead_shape, 3)
        dq = qexp(dt * omega)
        return qmul(q, dq)

    @torch.no_grad()
    def _update_spin_slots(self, q_query, indices):
        # q_query: (B,S,4), indices: (B,S,K)
        flat_idx = indices.reshape(-1)
        cur = self.q_memory_slots.index_select(0, flat_idx)
        q_rep = q_query.reshape(-1, 4).repeat_interleave(indices.size(-1), dim=0)
        upd = qnormalize(qmul(cur, q_rep))
        blended = qnormalize(0.9 * cur + 0.1 * upd)
        self.q_memory_slots.index_copy_(0, flat_idx, blended)

    def _blend_qhm_weights(self, q_feat, indices, w_geo):
        # q_feat: [N,D], indices: [N,K], w_geo: [N,K]
        if not self.qhm_enabled:
            return w_geo
        qhm = self.holo.weights(q_feat, indices)
        ctrl = self.lb_ctrl(qhm["resonance"], qhm["coherence"], qhm["weights"])
        w_geo_sharp = torch.softmax(
            torch.log(w_geo.clamp_min(1e-9)) / max(1e-6, float(ctrl["temp_mult"])),
            dim=-1,
        )
        if self.qhm_alpha_override is None:
            alpha = 0.5 * (qhm["alpha"] + self.qhm_alpha_head(q_feat)) + float(ctrl["alpha_boost"])
        else:
            alpha = torch.full_like(qhm["alpha"], float(self.qhm_alpha_override)) + float(
                ctrl["alpha_boost"]
            )
        alpha = alpha.clamp(0.0, float(ctrl["alpha_max"]))
        return (1 - alpha) * w_geo_sharp + alpha * qhm["weights"]

    @torch.no_grad()
    def holonomy_stats(self, manifold_x: torch.Tensor):
        # manifold_x: (B,S,D,3) or (B,D,3)
        if manifold_x.dim() == 4:
            patch = manifold_x.reshape(-1, manifold_x.size(-2), manifold_x.size(-1))
        elif manifold_x.dim() == 3:
            patch = manifold_x
        else:
            raise ValueError(
                f"Expected manifold_x with 3 or 4 dims, got {list(manifold_x.shape)}"
            )
        return self.hol_probe(patch)

    def _attend(self, query, positions):
        # query: (B,S,D) ; positions: (B,S,D,3)
        M = self.active_slots
        mem_pos = self.positional_encoding[:M].view(M, -1)     # (M,3D)
        q = positions.flatten(2).detach()                      # (B,S,3D)
        dist = fast_pairwise_l2(q, mem_pos)                    # (B,S,M)

        # Temperature scaling + curvature weight
        # Handle both scalar and per-sample temp
        if self.temperature.numel() == 1:
            temp_scale = self.temperature.clamp_min(1e-6)
        else:
            # Per-sample: (B,) -> (B,1,1) for broadcasting
            temp_scale = self.temperature.view(-1, 1, 1).clamp_min(1e-6)
        dist = dist / temp_scale
        # Use scalar curvature per slot to keep warping/broadcast stable.
        slot_curv = self.curvature_scalar[:M]
        curv_w = torch.exp(-self.curv_alpha * slot_curv.abs())  # (M,)
        dist = dist * curv_w.view(1,1,M)

        # Top-k with internal lightbulb boost
        K = min(self.K_base, M)
        dtmp, _ = torch.topk(dist, max(1,K), dim=-1, largest=False)
        top1 = dtmp[...,0].mean(dim=1).mean(dim=0) if dtmp.dim()==3 else dtmp.mean()
        self.lb_top1_avg = self.lb_momentum * self.lb_top1_avg + (1-self.lb_momentum) * top1.detach()
        internal_fire = bool(top1 < self.lb_drop_ratio * self.lb_top1_avg)
        if internal_fire:
            K = min(M, max(K, int(self.K_base * 1.5)))

        dtop, itop = torch.topk(dist, K, dim=-1, largest=False)        # (B,S,K)
        q_query = self._query_quat_from_manifold(positions)
        q_query = self._parallel_transport_spin(q_query, positions)
        q_feat = query.reshape(query.size(0) * query.size(1), self.D)
        dtop_flat = dtop.reshape(dtop.size(0) * dtop.size(1), K)
        itop_flat = itop.reshape(itop.size(0) * itop.size(1), K)
        curv_eff, mode_w4, mode_wext = self.geom_blender(q_feat)
        self.last_geom_weights = mode_wext
        topk_keys = self.metric_keys[:M][itop_flat]
        d_geo = self.geom_metric.distances(q_feat, topk_keys, mode_w4)
        dtop_mix = (1.0 - self.geom_gamma) * dtop_flat + self.geom_gamma * d_geo
        pre_probs = torch.softmax(-dtop_mix.detach(), dim=-1)
        pre_entropy = float((-(pre_probs * pre_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()).item())
        self.conformal_b = self.topology_v3.schedule_conformal_b(
            self.conformal_b,
            telemetry={"entropy": pre_entropy, "dist_mean": float(dtop_mix.detach().mean().item())},
        )
        dtop_warp, conformal_aux = warp_knn_with_stats(
            distances=dtop_mix,
            indices=itop_flat,
            query_vec=q_feat,
            curv_per_slot=curv_eff[:M],
            conformal_mlp=self.conformal_mlp,
            b=self.conformal_b,
        )
        qc = self.qc_head(q_feat)
        q_complex = (qc[:, : self.qc_dim] + 1j * qc[:, self.qc_dim :]).to(torch.complex64)
        mem_feat = self.positional_encoding[:M].mean(dim=-1)  # (M,D)
        self.geometry_merger.spd_L = self.spd_L
        w = self.geometry_merger(
            base_dist=dtop_warp,
            indices=itop_flat,
            q_feat=q_feat,
            mem_feat=mem_feat,
            slot_curv=curv_eff[:M],
            q_quat=q_query.reshape(q_query.size(0) * q_query.size(1), 4),
            mem_quat=self.q_memory_slots[:M],
            q_complex=q_complex,
            mem_complex=self.mem_complex[:M],
            return_weights=True,
        ).view_as(dtop)
        self.holo.renorm_slots()
        w = self._blend_qhm_weights(
            q_feat,
            itop_flat,
            w.reshape(w.size(0) * w.size(1), K),
        ).view_as(dtop)
        bsz, seq, _ = dtop.shape
        w_entropy = -(w * (w.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=1).detach()
        self.last_router_features = {
            "omega_mean": conformal_aux["omega_mean"].reshape(bsz, seq).mean(dim=1),
            "curv_mean": conformal_aux["curv_mean"].reshape(bsz, seq).mean(dim=1),
            "dist_mean": dtop_warp.reshape(bsz, seq, K).mean(dim=(1, 2)).detach(),
            "entropy": w_entropy,
        }
        mem = self.memory_slots[:M][itop]                              # (B,S,K,H)
        attended = torch.sum(w.unsqueeze(-1) * mem, dim=2)             # (B,S,H)
        return attended, (w, itop), internal_fire, q_query

    @torch.no_grad()
    def _record_usage(self, indices):
        flat = indices.reshape(-1)
        self.usage_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage_counts.dtype))

    @torch.no_grad()
    def _write(self, encoded, weights_indices, ema=0.9):
        w, idx = weights_indices
        # encoded: (B,S,H), w: (B,S,K), idx: (B,S,K)
        # Build per-(B,S,K) updates so source rows align 1:1 with flattened indices.
        flat_idx = idx.reshape(-1)
        flat_upd = (w.unsqueeze(-1) * encoded.unsqueeze(2)).reshape(-1, self.H)
        accum = torch.zeros_like(self.memory_slots)
        accum.index_add_(0, flat_idx, flat_upd)
        counts = torch.zeros(self.M, device=accum.device).index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=accum.dtype))
        counts = counts.clamp_min_(1.0).unsqueeze(-1)
        avg_upd = accum / counts
        self.memory_slots.mul_(ema).add_((1 - ema) * avg_upd)
        # DDP sync
        self._sync_buffers_ddp()

    def forward(self, x, operation='read', fire_mask=None, recall_boost: float = 0.3):
        """
        fire_mask: optional (B,) bool Tensor. If any True, we sharpen attention by lowering temperature
                   and we also write with a lower EMA (more plastic).
        """
        B,S,_ = x.shape

        # --- Lower temperature if any batch fires ---
        saved_temp = self.temperature.clone()
        any_ext_fire = (
            isinstance(fire_mask, torch.Tensor) and fire_mask.any()
        ) or (
            isinstance(fire_mask, bool) and fire_mask
        )
        if any_ext_fire:
            self.temperature = (self.temperature / (1.0 + recall_boost)).detach()

        man = self.manifold_projection(x).view(B,S,self.D,3)
        evolved = self._evolve(man)
        query = evolved.mean(dim=3)                                  # (B,S,D)
        attended, wi, internal_fire, q_query = self._attend(query, evolved)
        self._record_usage(wi[1])

        if operation == 'write':
            # More plastic on fire
            ema = 0.85 if (internal_fire or any_ext_fire) else 0.90
            enc = attended                                           # (B,S,H)
            self._write(enc, wi, ema=ema)
            self._update_spin_slots(q_query, wi[1])
            self.temperature = saved_temp
            return x

        out = self.output_projection(attended)                      # (B,S,input_dim)
        self.temperature = saved_temp
        return out
