import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .geometry_merger import GeometryMergerV3
from .geometry_utils import qexp, qmul, qnormalize
from .holo_head import HoloHead
from .lightbulb_controller import LightbulbController
from .memory.conformal import ConformalMLP, warp_knn_with_stats
from geometry.blend import GeometryBlender
from geometry.manifold_utils import product_et_distance, symplectic_leapfrog, wrap_angles
from topology.manager_v3 import DynamicTopologyManagerV2


class EnhancedCurvedMemory(nn.Module):
    """A simpler curved memory with scalar curvature gating and associative weights.
    Added usage tracking, active slots, consolidation, energy mode.
    Functions as working memory as well.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        curvature_dim: int = 8,
        mem_slots: int = 128,
        topk: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 0,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.H, self.M = hidden_dim, mem_slots
        self.K_base = min(topk, mem_slots)
        self.transformer_layers = int(max(0, transformer_layers))
        self.transformer_dropout = float(transformer_dropout)
        self.encoder = nn.Sequential(nn.Linear(input_dim, self.H), nn.Tanh())
        self.curvature = nn.Parameter(torch.randn(curvature_dim))
        self.curv_proj = nn.Linear(self.H, 1)  # scalar gate
        self.memory_slots = nn.Parameter(torch.randn(self.M, self.H))
        self.memory_importance = nn.Parameter(torch.ones(self.M))
        self.associative_weights = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        self.decoder = nn.Sequential(nn.Linear(self.H, input_dim), nn.Tanh())
        self.q_memory_slots = nn.Parameter(qnormalize(torch.randn(self.M, 4)))
        self.spin_conn = nn.Sequential(
            nn.Linear(self.H, 64), nn.SiLU(), nn.Linear(64, 3)
        )
        self.geometry_merger = GeometryMergerV3(
            q_dim=self.H, m_dim=self.H, spd_rank=4, use_heat_kernel=True
        )
        self.spd_L = nn.Parameter(torch.zeros(self.M, self.H, 4))
        self.geometry_merger.spd_L = self.spd_L
        self.memory_curvature = nn.Parameter(torch.zeros(self.M))
        self.qc_dim = 16
        self.mem_complex = nn.Parameter(
            (
                torch.randn(self.M, self.qc_dim)
                + 1j * torch.randn(self.M, self.qc_dim)
            ).to(torch.complex64)
        )
        self.qc_head = nn.Sequential(
            nn.Linear(self.H, 32), nn.SiLU(), nn.Linear(32, 2 * self.qc_dim)
        )
        self.holo = HoloHead(
            dim=self.H,
            num_slots=self.M,
            temperature=1.0,
            phase_noise_std=0.0,
            lightbulb_thresh=0.92,
            explosive_temp=0.55,
            explosive_alpha=0.65,
        )
        self.qhm_alpha_head = nn.Sequential(
            nn.Linear(self.H, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.lb_ctrl = LightbulbController(
            z_thresh_on=2.0,
            z_thresh_off=1.2,
            cooldown_steps=6,
            max_stage2_steps=2,
            budget_per_100=6,
            prefocus_temp_mult=0.9,
            explosive_temp_mult=0.55,
            prefocus_alpha_boost=0.12,
            explosive_alpha_boost=0.45,
        )
        self.qhm_enabled = True
        self.qhm_alpha_override = None
        self.curvature_scalar = nn.Parameter(torch.zeros(self.M))
        self.conformal_b = 0.1
        self.conformal_mlp = ConformalMLP(self.H)
        self.last_router_features = None
        self.geom_blender = GeometryBlender(num_slots=self.M)
        self.topology_v3 = DynamicTopologyManagerV2(bank="wm")
        self.topology = self.topology_v3
        self.last_geom_weights = None
        self.geom_gamma = 0.30
        self.use_product_manifold = True
        self.torus_k = max(2, min(8, self.H // 4))
        self.alpha_torus_base = 0.5
        self.split_proj_e = nn.Linear(self.H, self.H - self.torus_k, bias=False)
        self.split_proj_t = nn.Linear(self.H, self.torus_k, bias=False)
        self.h_q = nn.Linear(self.H, self.H)
        self.h_p = nn.Linear(self.H, self.H)
        self.symplectic_step = 1e-2
        self.enable_symplectic = False

        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M
        self.transformer_heads = int(transformer_heads) if int(transformer_heads) > 0 else self._pick_num_heads(self.input_dim)
        if self.transformer_layers > 0:
            self.input_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.input_dim,
                    nhead=self.transformer_heads,
                    dim_feedforward=max(128, self.input_dim * 2),
                    dropout=self.transformer_dropout,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=self.transformer_layers,
            )
            self.input_transformer_norm = nn.LayerNorm(self.input_dim)
        else:
            self.input_transformer = None
            self.input_transformer_norm = None
        self.external_attention_context = None
        self.external_memory_attn = nn.MultiheadAttention(
            self.input_dim,
            num_heads=self.transformer_heads,
            batch_first=True,
        )
        self.external_memory_attn_norm = nn.LayerNorm(self.input_dim)

    @staticmethod
    def _pick_num_heads(dim: int) -> int:
        for h in (8, 4, 2):
            if dim % h == 0:
                return h
        return 1

    def set_external_attention_context(self, context: torch.Tensor) -> None:
        if context is None:
            self.external_attention_context = None
            return
        ctx = torch.as_tensor(context, device=self.memory_slots.device, dtype=self.memory_slots.dtype)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0)
        elif ctx.dim() != 3:
            raise ValueError("external attention context must be [T,D], [B,D], or [B,S,D]")
        if ctx.size(-1) != self.input_dim:
            raise ValueError(f"external attention context dim must be {self.input_dim}")
        self.external_attention_context = ctx.detach()

    def clear_external_attention_context(self) -> None:
        self.external_attention_context = None

    # Controls
    def set_temperature(self, t: torch.Tensor):
        if t.numel() == 1:
            self.temperature = t.detach()
        else:
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
        curv_scalar = self.topology.curvature_scalar(self.curvature_scalar)
        _, wext = self.topology.mode_weights()
        curv_new = self.topology.mutate_curvature(curv_scalar, wext)
        self.curvature_scalar.copy_(curv_new)
        pri = self.topology.mode_priors().to(self.curvature_scalar.device)
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
        for t in [self.memory_slots, self.usage_counts, self.associative_weights]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()

    @torch.no_grad()
    def consolidate_unused(self, threshold: float = 0.1, ema: float = 0.9):
        if self.usage_counts.max() <= 0:
            return
        usage_ratio = self.usage_counts / (self.usage_counts.max() + 1e-6)
        mask = usage_ratio < threshold
        if mask.any():
            if (~mask).any():
                mean_slot = self.memory_slots[~mask].mean(dim=0, keepdim=True)
            else:
                mean_slot = self.memory_slots.mean(dim=0, keepdim=True)
            self.memory_slots[mask].copy_(ema * self.memory_slots[mask] + (1 - ema) * mean_slot)
            self.usage_counts[mask] = 0.0

    def get_metrics(self):
        """Return dict of diagnostic metrics."""
        m = self.active_slots
        out = {
            "curved_temp": self.temperature.mean().item()
            if self.temperature.numel() > 1
            else self.temperature.item(),
            "curved_topk_base": self.K_base,
            "curved_active_slots": m,
            "curved_usage_mean": self.usage_counts[:m].mean().item(),
            "curved_usage_max": self.usage_counts[:m].max().item(),
            "curved_importance_mean": self.memory_importance[:m].mean().item(),
            "curved_conformal_b": float(self.conformal_b),
            "curved_topology_fitness_ema": float(self.topology_v3.fitness_ema or 0.0),
        }
        if hasattr(self.topology_v3, "mode"):
            mode_map = {"hyperbolic": 0.0, "spherical": 1.0, "euclidean": 2.0, "fractal": 3.0}
            out["curved_topology_mode"] = mode_map.get(str(self.topology_v3.mode), -1.0)
        if isinstance(self.last_geom_weights, dict):
            for k in ("hyperbolic", "spherical", "euclidean", "fractal", "torus", "cp"):
                if k in self.last_geom_weights:
                    out[f"curved_geom_w_{k}"] = float(self.last_geom_weights[k])
        out["curved_transformer_layers"] = float(self.transformer_layers)
        return out

    # Core ops
    def content_based_addressing(self, query, return_dist: bool = False, slots=None):  # query: (B,H)
        m = self.active_slots
        if slots is None:
            slots = self.memory_slots[:m].detach().clone()
        qn = F.normalize(query, dim=-1)
        mem_n = F.normalize(slots, dim=-1)
        sim = torch.einsum("bd,md->bm", qn, mem_n)                        # (B,M)
        gate = torch.sigmoid(self.curv_proj(query))                       # (B,1)
        sim = sim * (0.5 + gate)                                          # scalar gating
        dist = (1.0 - sim) / self.temperature.clamp_min(1e-6)
        k = min(self.K_base, m)
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)
        if not return_dist:
            return vals, idx
        return vals, idx, vals

    def _query_quat_from_hidden(self, h):
        # h: (..., H) -> (..., 4)
        v = h[..., :3]
        ones = torch.ones_like(v[..., :1])
        return qnormalize(torch.cat([ones, 0.1 * v], dim=-1))

    def _parallel_transport_spin(self, q, h, dt=1.0):
        omega = self.spin_conn(h.reshape(-1, self.H)).view(*h.shape[:-1], 3)
        dq = qexp(dt * omega)
        return qmul(q, dq)

    @torch.no_grad()
    def _update_spin_slots(self, q_query, indices):
        # q_query: (B,4), indices: (B,K)
        flat_idx = indices.reshape(-1)
        cur = self.q_memory_slots.index_select(0, flat_idx)
        q_rep = q_query.repeat_interleave(indices.size(-1), dim=0)
        upd = qnormalize(qmul(cur, q_rep))
        blended = qnormalize(0.9 * cur + 0.1 * upd)
        self.q_memory_slots.index_copy_(0, flat_idx, blended)

    def _blend_qhm_weights(self, q_feat, indices, w_geo):
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
    def update_memory(self, x, importance, indices):
        enc = self.encoder(x).mean(dim=1)                                # (B,H)
        mem = self.memory_slots[indices]                                 # (B,K,H)
        gate = torch.sigmoid(self.memory_importance[indices].unsqueeze(-1))  # (B,K,1)
        cand = enc.unsqueeze(1)                                          # (B,1,H)
        upd = gate * mem + (1 - gate) * cand                             # (B,K,H)
        flat_idx = indices.reshape(-1)
        flat_upd = upd.reshape(-1, self.H)
        accum = torch.zeros_like(self.memory_slots)
        accum.index_add_(0, flat_idx, flat_upd)
        counts = torch.zeros(self.M, device=accum.device).index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=accum.dtype)
        )
        counts = counts.clamp_min_(1.0).unsqueeze(-1)
        avg_upd = accum / counts
        self.memory_slots.mul_(0.95).add_(0.05 * avg_upd)
        self._sync_buffers_ddp()
        # importance EMA (crude)
        self.memory_importance.index_add_(
            0,
            flat_idx,
            0.05
            * torch.ones_like(flat_idx, dtype=self.memory_importance.dtype).to(
                self.memory_importance.device
            ),
        )

    @torch.no_grad()
    def _record_usage(self, indices):
        flat = indices.reshape(-1)
        self.usage_counts.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage_counts.dtype))

    def associative_activation(self, query, slots=None):  # (B,H) -> (B,M)
        m = self.active_slots
        if slots is None:
            slots = self.memory_slots[:m].detach().clone()
        act = torch.einsum("bd,md->bm", query, slots)
        assoc = self.associative_weights[:m, :m].detach().clone()
        for _ in range(2):
            act = torch.softmax(act, dim=-1)
            act = torch.einsum("bm,mn->bn", act, assoc)
        return act

    def forward(self, x, operation="read", importance=None):
        x_in = x
        if self.input_transformer is not None:
            tx = self.input_transformer(x_in)
            x_in = self.input_transformer_norm(x_in + tx)
        if self.external_attention_context is not None:
            ctx = self.external_attention_context.to(device=x_in.device, dtype=x_in.dtype)
            if ctx.size(0) == 1 and x_in.size(0) > 1:
                ctx = ctx.expand(x_in.size(0), -1, -1)
            elif ctx.size(0) != x_in.size(0):
                ctx = ctx.mean(dim=0, keepdim=True).expand(x_in.size(0), -1, -1)
            x_ext, _ = self.external_memory_attn(x_in, ctx, ctx, need_weights=False)
            x_in = self.external_memory_attn_norm(x_in + x_ext)
        enc = self.encoder(x_in)                                            # (B,S,H)
        if self.enable_symplectic:
            q_dyn = enc
            p_dyn = torch.zeros_like(enc)
            dH_dq = self.h_q(q_dyn)
            dH_dp = self.h_p(p_dyn)
            q_dyn, _ = symplectic_leapfrog(q_dyn, p_dyn, dH_dq, dH_dp, step=self.symplectic_step)
            query = q_dyn.mean(dim=1)
        else:
            query = enc.mean(dim=1)                                      # (B,H)
        q_query = self._query_quat_from_hidden(query)
        q_query = self._parallel_transport_spin(q_query, query)
        if operation == "write":
            _, idx = self.content_based_addressing(query)
            self._record_usage(idx)
            self.update_memory(x_in, importance, idx)
            self._update_spin_slots(q_query, idx)
            return x
        m = self.active_slots
        read_slots = self.memory_slots[:m].detach().clone()
        vals, idx, dist_top = self.content_based_addressing(query, return_dist=True, slots=read_slots)  # (B,K)
        curv_eff, _, mode_wext = self.geom_blender(query)
        self.last_geom_weights = mode_wext
        torus_w = float(mode_wext.get("torus", 0.0))
        alpha_torus = self.alpha_torus_base * (0.75 + 0.5 * torus_w)
        if self.use_product_manifold:
            topk_keys = read_slots[idx]  # (B,K,H)
            q_e = self.split_proj_e(query)
            q_t = wrap_angles(self.split_proj_t(query))
            k_e = self.split_proj_e(topk_keys)
            k_t = wrap_angles(self.split_proj_t(topk_keys))
            d_geo = product_et_distance(q_e, q_t, k_e, k_t, alpha=alpha_torus)
            dist_top = (1.0 - self.geom_gamma) * dist_top + self.geom_gamma * d_geo
        pre_probs = torch.softmax(-dist_top.detach(), dim=-1)
        pre_entropy = float((-(pre_probs * pre_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()).item())
        self.conformal_b = self.topology_v3.schedule_conformal_b(
            self.conformal_b,
            telemetry={"entropy": pre_entropy, "dist_mean": float(dist_top.detach().mean().item())},
        )
        dist_top, conformal_aux = warp_knn_with_stats(
            distances=dist_top,
            indices=idx,
            query_vec=query,
            curv_per_slot=curv_eff[: self.active_slots],
            conformal_mlp=self.conformal_mlp,
            b=self.conformal_b,
        )
        qc = self.qc_head(query)
        q_complex = (qc[:, : self.qc_dim] + 1j * qc[:, self.qc_dim :]).to(torch.complex64)
        self.geometry_merger.spd_L = self.spd_L
        geom_w = self.geometry_merger(
            base_dist=dist_top,
            indices=idx,
            q_feat=query,
            mem_feat=read_slots,
            slot_curv=self.memory_curvature[:m],
            q_quat=q_query,
            mem_quat=self.q_memory_slots[:m],
            q_complex=q_complex,
            mem_complex=self.mem_complex[:m],
            return_weights=True,
        )
        self.holo.renorm_slots()
        geom_w = self._blend_qhm_weights(query, idx, geom_w)
        w_entropy = -(geom_w * (geom_w.clamp_min(1e-9)).log()).sum(dim=-1).detach()
        self.last_router_features = {
            "omega_mean": conformal_aux["omega_mean"],
            "curv_mean": conformal_aux["curv_mean"],
            "dist_mean": dist_top.mean(dim=-1).detach(),
            "entropy": w_entropy,
        }
        self._record_usage(idx)
        act = self.associative_activation(query, slots=read_slots)       # (B,M_active)
        comb = torch.softmax(torch.log(geom_w.clamp_min(1e-9)) + act.gather(1, idx), dim=-1)
        mem = read_slots[idx]                                            # (B,K,H)
        read = torch.sum(comb.unsqueeze(-1) * mem, dim=1)               # (B,H)
        return self.decoder(read).unsqueeze(1).expand(-1, x.size(1), -1)  # (B,S,input_dim)

    # --------- Bridge adapters (TripleHybrid / HG Episodic LTM) ----------
    @torch.no_grad()
    def ingest_external_vectors(self, vectors: torch.Tensor, *, write_scale: float = 1.0) -> None:
        """
        Ingest external episodic vectors into curved memory.
        Accepts [T,D], [B,D], or [B,S,D] tensors in input_dim space.
        """
        x = torch.as_tensor(vectors, device=self.memory_slots.device, dtype=self.memory_slots.dtype)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1,T,D]
        elif x.dim() != 3:
            raise ValueError("vectors must be [T,D], [B,D], or [B,S,D]")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"vectors last dim must be input_dim={self.input_dim}")
        if not torch.isfinite(x).all():
            raise ValueError("vectors contains NaN/Inf")
        if float(write_scale) != 1.0:
            x = x * float(write_scale)
        _ = self.forward(x, operation="write", importance=None)

    def read_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper used by TripleHybrid bridge helpers."""
        return self.forward(x, operation="read")
