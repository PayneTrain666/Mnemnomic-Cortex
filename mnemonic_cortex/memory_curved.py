import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .geometry_merger import GeometryMergerV3
from .geometry_utils import qexp, qmul, qnormalize
from .holo_head import HoloHead
from .lightbulb_controller import LightbulbController


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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.H, self.M = hidden_dim, mem_slots
        self.K_base = min(topk, mem_slots)
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

        self.register_buffer("temperature", torch.tensor(1.0))

        # Usage + active slots
        self.register_buffer("usage_counts", torch.zeros(self.M, dtype=torch.float32))
        self.active_slots = self.M

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

    # ---------- DDP sync -------------
    @torch.no_grad()
    def _sync_buffers_ddp(self):
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        for t in [self.memory_slots.data, self.usage_counts, self.associative_weights.data]:
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
            self.memory_slots.data[mask] = ema * self.memory_slots.data[mask] + (1 - ema) * mean_slot
            self.usage_counts[mask] = 0.0

    def get_metrics(self):
        """Return dict of diagnostic metrics."""
        m = self.active_slots
        return {
            "curved_temp": self.temperature.mean().item()
            if self.temperature.numel() > 1
            else self.temperature.item(),
            "curved_topk_base": self.K_base,
            "curved_active_slots": m,
            "curved_usage_mean": self.usage_counts[:m].mean().item(),
            "curved_usage_max": self.usage_counts[:m].max().item(),
            "curved_importance_mean": self.memory_importance[:m].mean().item(),
        }

    # Core ops
    def content_based_addressing(self, query):  # query: (B,H)
        m = self.active_slots
        qn = F.normalize(query, dim=-1)
        mem_n = F.normalize(self.memory_slots[:m], dim=-1)
        sim = torch.einsum("bd,md->bm", qn, mem_n)                        # (B,M)
        gate = torch.sigmoid(self.curv_proj(query))                       # (B,1)
        sim = sim * (0.5 + gate)                                          # scalar gating
        dist = (1.0 - sim) / self.temperature.clamp_min(1e-6)
        k = min(self.K_base, m)
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)
        return vals, idx

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
        self.q_memory_slots.data.index_copy_(0, flat_idx, blended)

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
        self.memory_slots.data.mul_(0.95).add_(0.05 * avg_upd)
        self._sync_buffers_ddp()
        # importance EMA (crude)
        self.memory_importance.data.index_add_(
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

    def associative_activation(self, query):  # (B,H) -> (B,M)
        m = self.active_slots
        act = torch.einsum("bd,md->bm", query, self.memory_slots[:m])
        for _ in range(2):
            act = torch.softmax(act, dim=-1)
            act = torch.einsum("bm,mn->bn", act, self.associative_weights[:m, :m])
        return act

    def forward(self, x, operation="read", importance=None):
        enc = self.encoder(x)                                            # (B,S,H)
        query = enc.mean(dim=1)                                          # (B,H)
        q_query = self._query_quat_from_hidden(query)
        q_query = self._parallel_transport_spin(q_query, query)
        if operation == "write":
            _, idx = self.content_based_addressing(query)
            self._record_usage(idx)
            self.update_memory(x, importance, idx)
            self._update_spin_slots(q_query, idx)
            return x
        vals, idx = self.content_based_addressing(query)                 # (B,K)
        qc = self.qc_head(query)
        q_complex = (qc[:, : self.qc_dim] + 1j * qc[:, self.qc_dim :]).to(torch.complex64)
        m = self.active_slots
        self.geometry_merger.spd_L = self.spd_L
        geom_w = self.geometry_merger(
            base_dist=vals,
            indices=idx,
            q_feat=query,
            mem_feat=self.memory_slots[:m],
            slot_curv=self.memory_curvature[:m],
            q_quat=q_query,
            mem_quat=self.q_memory_slots[:m],
            q_complex=q_complex,
            mem_complex=self.mem_complex[:m],
            return_weights=True,
        )
        self.holo.renorm_slots()
        geom_w = self._blend_qhm_weights(query, idx, geom_w)
        self._record_usage(idx)
        act = self.associative_activation(query)                         # (B,M_active)
        comb = torch.softmax(torch.log(geom_w.clamp_min(1e-9)) + act.gather(1, idx), dim=-1)
        mem = self.memory_slots[idx]                                     # (B,K,H)
        read = torch.sum(comb.unsqueeze(-1) * mem, dim=1)               # (B,H)
        return self.decoder(read).unsqueeze(1).expand(-1, x.size(1), -1)  # (B,S,input_dim)
