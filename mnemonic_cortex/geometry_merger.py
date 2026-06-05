import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_utils import (
    fubini_study_batched,
    lorentz_dist,
    lorentz_lift,
    qangle,
    qnormalize,
    sph_warp,
)


class GeometryMergerV3(nn.Module):
    """
    Like V2, but with:
      - Conformal scalar omega(x): distance -> exp(omega) * distance (bounded)
      - Small clamped micro-warp b
      - Backward-compatible debug keys
    """

    def __init__(
        self,
        q_dim: int,
        m_dim: int,
        spd_rank: int = 4,
        use_heat_kernel: bool = True,
        micro_b: float = 0.02,
        omega_max: float = 0.20,
    ):
        super().__init__()
        self.use_heat = use_heat_kernel

        self.spd_rank = spd_rank
        self.spd_L = None
        self.spd_eps = 1e-3

        self.register_buffer("micro_b", torch.tensor(float(micro_b)))
        self.register_buffer("micro_b_max", torch.tensor(0.05))
        self.register_buffer("omega_max", torch.tensor(float(omega_max)))
        self.register_buffer("temp_scale", torch.tensor(1.0))

        self.omega_head = nn.Sequential(
            nn.Linear(q_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self.gate = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1),
        )
        self.register_buffer("gate_bias", torch.zeros(6))

        self.t_head = nn.Sequential(
            nn.Linear(q_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

        self.spin_scale = nn.Parameter(torch.tensor(0.7))

        self._last_debug = {
            "w_avg": None,
            "t_avg": None,
            "phi_scale_avg": None,
            "omega_avg": None,
        }

    @torch.no_grad()
    def set_gate_bias(self, bias_vec: torch.Tensor):
        self.gate_bias.copy_(bias_vec.to(self.gate_bias.device, self.gate_bias.dtype))

    @torch.no_grad()
    def set_temp_scale(self, s: float):
        self.temp_scale.fill_(float(s))

    @torch.no_grad()
    def set_micro_b(self, b: float):
        b = min(float(b), float(self.micro_b_max.item()))
        self.micro_b.fill_(b)

    @torch.no_grad()
    def set_omega_max(self, max_abs: float):
        self.omega_max.fill_(float(max_abs))

    def spd_distance(self, q_feat: torch.Tensor, mem_feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        bsz, dim = q_feat.shape
        k = indices.size(1)
        m_sel = mem_feat.index_select(0, indices.reshape(-1)).view(bsz, k, dim)

        if self.spd_L is None:
            return torch.norm(q_feat.unsqueeze(1) - m_sel, dim=-1)

        l_sel = self.spd_L.index_select(0, indices.reshape(-1)).view(bsz, k, dim, self.spd_rank)
        diff = (q_feat.unsqueeze(1) - m_sel).unsqueeze(-1)
        llt = torch.matmul(l_sel, l_sel.transpose(-1, -2))
        eye = torch.eye(dim, device=q_feat.device, dtype=q_feat.dtype).view(1, 1, dim, dim)
        g = llt + self.spd_eps * eye
        gdiff = torch.matmul(torch.matmul(diff.transpose(-2, -1), g), diff).squeeze(-1).squeeze(-1)
        return torch.sqrt(gdiff.clamp_min(1e-12))

    def forward(
        self,
        base_dist: torch.Tensor,
        indices: torch.Tensor,
        q_feat: torch.Tensor,
        mem_feat: torch.Tensor,
        slot_curv: torch.Tensor,
        q_quat: torch.Tensor = None,
        mem_quat: torch.Tensor = None,
        q_complex: torch.Tensor = None,
        mem_complex: torch.Tensor = None,
        return_weights: bool = True,
    ) -> torch.Tensor:
        bsz, k = base_dist.shape

        if slot_curv.dim() == 2:
            slot_c_all = slot_curv.mean(dim=1)
        else:
            slot_c_all = slot_curv
        slot_c = slot_c_all.index_select(0, indices.reshape(-1)).view(bsz, k)

        d_euc = base_dist
        d_spd = self.spd_distance(q_feat, mem_feat, indices)
        d_sph = sph_warp(base_dist, gamma=0.9)

        q_l = lorentz_lift(q_feat)
        m_sel = mem_feat.index_select(0, indices.reshape(-1)).view(bsz, k, -1)
        m_l = lorentz_lift(m_sel)
        d_lor = lorentz_dist(q_l.unsqueeze(1), m_l)

        if (q_quat is not None) and (mem_quat is not None):
            qq = qnormalize(q_quat)
            mq = qnormalize(mem_quat.index_select(0, indices.reshape(-1))).view(bsz, k, 4)
            d_spin = self.spin_scale * (qangle(qq.unsqueeze(1).expand(bsz, k, 4), mq) / math.pi)
        else:
            d_spin = torch.zeros_like(base_dist)

        if (q_complex is not None) and (mem_complex is not None):
            mc = mem_complex.index_select(0, indices.reshape(-1)).view(bsz, k, -1)
            d_fs = fubini_study_batched(q_complex, mc)
        else:
            d_fs = torch.zeros_like(base_dist)

        with torch.no_grad():
            s0 = q_feat.abs().mean(dim=-1)
            s1 = q_feat.std(dim=-1)
            s2 = slot_c.abs().mean(dim=-1)
            s3 = d_euc.mean(dim=-1)
            s4 = d_spd.mean(dim=-1)
            s5 = d_lor.mean(dim=-1)
            gate_in = torch.stack([s0, s1, s2, s3, s4, s5], dim=-1)

        w_int = self.gate(gate_in)
        if torch.any(self.gate_bias != 0):
            logits = (w_int + 1e-6).log() + self.gate_bias.view(1, -1)
            w = torch.softmax(logits, dim=-1)
        else:
            w = w_int

        d_blend = (
            w[:, 0:1] * d_euc
            + w[:, 1:2] * d_spd
            + w[:, 2:3] * d_lor
            + w[:, 3:4] * d_sph
            + w[:, 4:5] * d_spin
            + w[:, 5:6] * d_fs
        )

        b = self.micro_b.clamp_max(self.micro_b_max)
        d_blend = d_blend * (1.0 + b * torch.tanh(slot_c) * d_blend)

        omega = self.omega_head(q_feat).clamp(-self.omega_max, self.omega_max)
        conf_scale = torch.exp(omega)
        d_final = d_blend * conf_scale

        if self.use_heat:
            t = self.t_head(q_feat).clamp_min(1e-3) * self.temp_scale.clamp(0.25, 4.0)
            kx = torch.exp(-(d_final**2) / (4.0 * t))
            weights = kx / (kx.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            t = torch.ones_like(conf_scale)
            weights = F.softmax(-d_final, dim=-1)

        with torch.no_grad():
            self._last_debug = {
                "w_avg": w.mean(dim=0).detach().cpu(),
                "t_avg": t.mean().item(),
                "phi_scale_avg": conf_scale.mean().item(),
                "omega_avg": omega.mean().item(),
            }

        return weights if return_weights else d_final


# Backward-compat aliases.
GeometryMergerV2 = GeometryMergerV3
GeometryMerger = GeometryMergerV3
