"""
Plain-language summary
----------------------
What this file is for: Knowledge-distillation helpers across domains.
How it fits in the system: Lets one part of the system teach another.
Status: OPT-IN
Important notes for non-coders: Training technique, not a memory bank.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distill_loss(a: torch.Tensor, b: torch.Tensor):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1).mean()


def kl_dist(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    p = (p + eps) / (p.sum(dim=-1, keepdim=True) + eps)
    q = (q + eps) / (q.sum(dim=-1, keepdim=True) + eps)
    return (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=-1).mean()


class CrossDomainDistiller(nn.Module):
    def __init__(self, multi_cps, cms=None, neighbor_k: int = 16, sim_temp: float = 0.07):
        super().__init__()
        self.multi_cps = multi_cps
        self.cms = cms
        self.neighbor_k = int(neighbor_k)
        self.sim_temp = float(sim_temp)

    @torch.no_grad()
    def _fused_cps(self, domain: str, key: str):
        cps, fuser = self.multi_cps.get(domain)
        fused, _ = fuser.fuse(cps.get(key).view())
        return fused

    @torch.no_grad()
    def _fused_cms(self, key: str):
        if self.cms is None or key not in self.cms.keys():
            return None
        return self.cms.read(key)

    def embedding_distill(self, teacher_domain: str, student_domain: str, keys: List[str], w_cos: float = 1.0, w_mse: float = 0.2):
        pairs = [(k, k) for k in keys]
        if not pairs:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        t = torch.stack([self._fused_cps(teacher_domain, k1) for k1, _ in pairs], dim=0)
        s = torch.stack([self._fused_cps(student_domain, k2) for _, k2 in pairs], dim=0)
        return float(w_cos) * cosine_distill_loss(t, s) + float(w_mse) * F.mse_loss(s, t)

    def cms_teacher_distill(self, student_domain: str, keys: List[str], w_cos: float = 1.0, w_mse: float = 0.2):
        if self.cms is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        t_list = []
        s_list = []
        for k in keys:
            t = self._fused_cms(k)
            if t is None:
                continue
            s = self._fused_cps(student_domain, k)
            t_list.append(t)
            s_list.append(s)
        if not t_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        t = torch.stack(t_list, dim=0)
        s = torch.stack(s_list, dim=0)
        return float(w_cos) * cosine_distill_loss(t, s) + float(w_mse) * F.mse_loss(s, t)

