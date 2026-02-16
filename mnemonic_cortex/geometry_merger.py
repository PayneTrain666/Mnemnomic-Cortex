import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_utils import qangle, qnormalize


def sph_warp(dist: torch.Tensor, gamma: float = 0.9) -> torch.Tensor:
    return (2.0 * torch.sin(0.5 * gamma * dist)).abs()


def lorentz_lift(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [..., D] -> [..., D+1], on upper sheet hyperboloid.
    sq = torch.sum(x * x, dim=-1, keepdim=True)
    t = torch.sqrt(1.0 + sq + eps)
    return torch.cat([t, x], dim=-1)


def lorentz_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Minkowski inner product: <a,b>_L = -a0*b0 + sum_i ai*bi
    ip = -(a[..., :1] * b[..., :1]).sum(dim=-1) + (a[..., 1:] * b[..., 1:]).sum(dim=-1)
    z = (-ip).clamp_min(1.0 + eps)
    return torch.acosh(z)


def fubini_study(q_complex: torch.Tensor, mem_complex: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q_complex: [B,C], mem_complex: [B,K,C] -> [B,K]
    qn = q_complex / (q_complex.norm(dim=-1, keepdim=True).clamp_min(eps))
    mn = mem_complex / (mem_complex.norm(dim=-1, keepdim=True).clamp_min(eps))
    overlap = (qn.unsqueeze(1).conj() * mn).sum(dim=-1).abs().clamp(0.0, 1.0 - eps)
    return torch.acos(overlap)


class GeometryMergerV2(nn.Module):
    """
    Mix Euclidean / SPD / Lorentz / spherical / spin / Fubini-Study channels.
    """

    def __init__(self, q_dim: int, m_dim: int, spd_rank: int = 4, use_heat_kernel: bool = True):
        super().__init__()
        self.use_heat = use_heat_kernel
        self.spd_rank = spd_rank
        self.spd_L = None  # assigned by owner memory module
        self.spd_eps = 1e-3

        self.gate = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1),
        )

        self.t_head = nn.Sequential(
            nn.Linear(q_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        self.phi = nn.Sequential(
            nn.Linear(q_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.spin_scale = nn.Parameter(torch.tensor(0.7))
        self.local_beta = 0.10

    def spd_distance(self, q_feat: torch.Tensor, mem_feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # q_feat: [B,D], mem_feat:[M,D], indices:[B,K] -> [B,K]
        bsz, dim = q_feat.shape
        k = indices.size(1)
        m_sel = mem_feat.index_select(0, indices.reshape(-1)).view(bsz, k, dim)

        if self.spd_L is None:
            return torch.norm(q_feat.unsqueeze(1) - m_sel, dim=-1)

        l_sel = self.spd_L.index_select(0, indices.reshape(-1)).view(bsz, k, dim, self.spd_rank)
        diff = (q_feat.unsqueeze(1) - m_sel).unsqueeze(-1)  # [B,K,D,1]
        llt = torch.matmul(l_sel, l_sel.transpose(-1, -2))  # [B,K,D,D]
        eye = torch.eye(dim, device=q_feat.device, dtype=q_feat.dtype).view(1, 1, dim, dim)
        g = llt + self.spd_eps * eye
        qf = torch.matmul(torch.matmul(diff.transpose(-2, -1), g), diff).squeeze(-1).squeeze(-1)
        return torch.sqrt(qf.clamp_min(1e-12))

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
        d_sph = sph_warp(base_dist, gamma=0.9)
        d_spd = self.spd_distance(q_feat, mem_feat, indices)

        q_l = lorentz_lift(q_feat)  # [B,D+1]
        m_sel = mem_feat.index_select(0, indices.reshape(-1)).view(bsz, k, -1)
        m_l = lorentz_lift(m_sel)  # [B,K,D+1]
        d_lor = lorentz_dist(q_l.unsqueeze(1).expand_as(m_l), m_l)

        if (q_quat is not None) and (mem_quat is not None):
            qq = qnormalize(q_quat)
            mq = qnormalize(mem_quat.index_select(0, indices.reshape(-1))).view(bsz, k, 4)
            d_spin = self.spin_scale * (qangle(qq.unsqueeze(1).expand(bsz, k, 4), mq) / math.pi)
        else:
            d_spin = torch.zeros_like(base_dist)

        if (q_complex is not None) and (mem_complex is not None):
            mc = mem_complex.index_select(0, indices.reshape(-1)).view(bsz, k, -1)
            d_fs = fubini_study(q_complex, mc)
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

        w = self.gate(gate_in)  # [B,6]
        d_blend = (
            w[:, 0:1] * d_euc
            + w[:, 1:2] * d_spd
            + w[:, 2:3] * d_lor
            + w[:, 3:4] * d_sph
            + w[:, 4:5] * d_spin
            + w[:, 5:6] * d_fs
        )

        d_blend = d_blend * (1.0 + self.local_beta * torch.tanh(slot_c) * d_blend)
        scale = torch.exp(0.5 * self.phi(q_feat).clamp(-2, 2))
        d_final = d_blend * scale

        if not return_weights:
            return d_final

        if self.use_heat:
            t = self.t_head(q_feat).clamp(1e-3, 0.3)
            kx = torch.exp(-(d_final**2) / (4.0 * t))
            weights = kx / (kx.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            weights = F.softmax(-d_final, dim=-1)
        return weights


# Backward-compat alias for previous imports.
GeometryMerger = GeometryMergerV2
