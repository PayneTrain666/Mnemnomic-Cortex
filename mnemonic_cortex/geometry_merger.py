import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_utils import _d_euc, _d_hyp, _d_sph, qangle, qnormalize

class GeometryMerger(nn.Module):
    """
    Mix Euclidean/Hyperbolic/Spherical/Spin distances; add gentle local curvature warp;
    optionally treat spin as a warped-product factor.
    Call this right before attention softmax.
    """
    def __init__(self, init_tau=10.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(4, 32), nn.SiLU(),
                                  nn.Linear(32, 4), nn.Softmax(-1))
        self.tau = nn.Parameter(torch.tensor([init_tau]*4, dtype=torch.float32))
        self.spin_scale = nn.Parameter(torch.tensor(0.7))
        self.local_beta = 0.12  # small curvature micro-warp

    def forward(self, dist, indices, slot_curv, q_query=None, q_memory=None, context_stat=None):
        """
        dist:     [B,K]
        indices:  [B,K]
        slot_curv:[mem_slots] or [mem_slots,D] (we mean-reduce to [mem_slots])
        q_query:  [B,4] or None
        q_memory: [mem_slots,4] or None
        context_stat: optional scalar tensor (B,) or None (used as a simple gate input)
        """
        B, K = dist.shape
        device, dtype = dist.device, dist.dtype

        if slot_curv.dim() == 2:
            slot_c_all = slot_curv.mean(dim=1)    # [mem_slots]
        else:
            slot_c_all = slot_curv                # [mem_slots]
        slot_c = slot_c_all.index_select(0, indices.view(-1)).view(B, K)

        # components
        de = _d_euc(dist)
        dh = _d_hyp(dist, slot_c, alpha=0.6)
        ds = _d_sph(dist, gamma=0.9)

        if (q_query is not None) and (q_memory is not None):
            q_query = qnormalize(q_query)
            q_mem = qnormalize(q_memory.index_select(0, indices.view(-1))).view(B,K,4)
            dsp = (self.spin_scale * (qangle(q_query.unsqueeze(1).expand(B,K,4), q_mem)/math.pi))
        else:
            dsp = torch.zeros_like(dist)

        # gate stats (simple and stable)
        with torch.no_grad():
            p = F.softmax(-dist, dim=-1)
            entropy = (-p * (p.clamp_min(1e-9)).log()).sum(-1).mean()
            mean_abs_c = slot_c.abs().mean()
            spin_coh = 1.0 - dsp.mean()
            if context_stat is None:
                context_stat = torch.full((B,), 0.5, device=device, dtype=dtype)
            stats = torch.tensor([entropy.item(), mean_abs_c.item(), spin_coh.item(), context_stat.mean().item()],
                                 device=device).unsqueeze(0)  # [1,4]

        w = self.gate(stats).squeeze(0)  # [4]

        # blended distance
        d_mix = (w[0]*de + w[1]*dh + w[2]*ds + w[3]*dsp)
        d_mix = d_mix * (1.0 + self.local_beta * torch.tanh(slot_c) * d_mix)  # micro-warp

        # optional warped-product (spin as second factor)
        d_final = torch.sqrt(d_mix**2 + 0.5 * dsp**2)

        return d_final