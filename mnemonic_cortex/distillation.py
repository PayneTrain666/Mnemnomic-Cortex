from dataclasses import dataclass, field
from typing import List, Optional, Sequence

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


@dataclass
class DistillationConfig:
    enabled: bool = False
    embedding_weight: float = 0.5
    mse_weight: float = 0.1
    neighbor_kl_weight: float = 0.2
    cms_teacher_weight: float = 0.2
    teacher_domain: str = "core"
    student_domains: Sequence[str] = field(default_factory=lambda: ("reasoning",))
    neighbor_k: int = 16
    sim_temp: float = 0.07

    def validate(self) -> None:
        if self.embedding_weight < 0.0:
            raise ValueError("embedding_weight must be non-negative")
        if self.mse_weight < 0.0:
            raise ValueError("mse_weight must be non-negative")
        if self.neighbor_kl_weight < 0.0:
            raise ValueError("neighbor_kl_weight must be non-negative")
        if self.cms_teacher_weight < 0.0:
            raise ValueError("cms_teacher_weight must be non-negative")
        if int(self.neighbor_k) <= 0:
            raise ValueError("neighbor_k must be positive")
        if float(self.sim_temp) <= 0.0:
            raise ValueError("sim_temp must be positive")
        if not str(self.teacher_domain):
            raise ValueError("teacher_domain must be non-empty")
        if not self.student_domains:
            raise ValueError("student_domains must be non-empty")


class CrossDomainDistiller(nn.Module):
    def __init__(self, multi_cps, cms=None, neighbor_k: int = 16, sim_temp: float = 0.07):
        super().__init__()
        self.multi_cps = multi_cps
        self.cms = cms
        self.neighbor_k = int(neighbor_k)
        self.sim_temp = float(sim_temp)
        self.distill_config = DistillationConfig(
            enabled=False,
            neighbor_k=self.neighbor_k,
            sim_temp=self.sim_temp,
        )

    def configure(self, cfg: DistillationConfig):
        cfg.validate()
        self.distill_config = cfg
        self.neighbor_k = int(cfg.neighbor_k)
        self.sim_temp = float(cfg.sim_temp)
        return self

    def _zero(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        return torch.tensor(0.0, device=device if device is not None else "cpu", dtype=dtype or torch.float32)

    def _infer_device_dtype(self):
        modules = []
        if self.multi_cps is not None:
            modules.append(self.multi_cps)
        if self.cms is not None:
            modules.append(self.cms)
        for module in modules:
            for p in module.parameters():
                return p.device, p.dtype
        return None, None

    @torch.no_grad()
    def _fused_cps(self, domain: str, key: str):
        if self.multi_cps is None or not self.multi_cps.has_domain(domain):
            return None
        cps, fuser = self.multi_cps.get(domain)
        if key not in cps._registry:
            return None
        fused, _ = fuser.fuse(cps.get(key).view())
        return fused

    @torch.no_grad()
    def _fused_cms(self, key: str):
        if self.cms is None or key not in self.cms.keys():
            return None
        return self.cms.read(key)

    def embedding_distill(self, teacher_domain: str, student_domain: str, keys: List[str], w_cos: float = 1.0, w_mse: float = 0.2):
        device, dtype = self._infer_device_dtype()
        t_list = []
        s_list = []
        for key in keys:
            t = self._fused_cps(teacher_domain, key)
            s = self._fused_cps(student_domain, key)
            if t is None or s is None:
                continue
            t_list.append(t)
            s_list.append(s)
        if not t_list:
            return self._zero(device=device, dtype=dtype)
        t = torch.stack(t_list, dim=0)
        s = torch.stack(s_list, dim=0)
        return float(w_cos) * cosine_distill_loss(t, s) + float(w_mse) * F.mse_loss(s, t)

    def cms_teacher_distill(self, student_domain: str, keys: List[str], w_cos: float = 1.0, w_mse: float = 0.2):
        device, dtype = self._infer_device_dtype()
        if self.cms is None:
            return self._zero(device=device, dtype=dtype)
        t_list = []
        s_list = []
        for k in keys:
            t = self._fused_cms(k)
            if t is None:
                continue
            s = self._fused_cps(student_domain, k)
            if s is None:
                continue
            t_list.append(t)
            s_list.append(s)
        if not t_list:
            return self._zero(device=device, dtype=dtype)
        t = torch.stack(t_list, dim=0)
        s = torch.stack(s_list, dim=0)
        return float(w_cos) * cosine_distill_loss(t, s) + float(w_mse) * F.mse_loss(s, t)

    def neighbor_kl_distill(self, teacher_domain: str, student_domain: str, keys: List[str], k: Optional[int] = None):
        device, dtype = self._infer_device_dtype()
        t_list = []
        s_list = []
        for key in keys:
            t = self._fused_cps(teacher_domain, key)
            s = self._fused_cps(student_domain, key)
            if t is None or s is None:
                continue
            t_list.append(t)
            s_list.append(s)
        if len(t_list) < 2:
            return self._zero(device=device, dtype=dtype)
        t = F.normalize(torch.stack(t_list, dim=0), dim=-1)
        s = F.normalize(torch.stack(s_list, dim=0), dim=-1)
        temp = max(1e-6, float(self.sim_temp))
        t_sim = (t @ t.T) / temp
        s_sim = (s @ s.T) / temp
        # remove self-neighbor diagonal
        eye = torch.eye(t_sim.size(0), device=t_sim.device, dtype=torch.bool)
        t_sim = t_sim.masked_fill(eye, float("-inf"))
        s_sim = s_sim.masked_fill(eye, float("-inf"))
        kk = int(max(1, min(int(k or self.neighbor_k), t_sim.size(0) - 1)))
        topi = torch.topk(t_sim, k=kk, dim=-1).indices
        t_rows = torch.gather(t_sim, 1, topi)
        s_rows = torch.gather(s_sim, 1, topi)
        t_prob = torch.softmax(t_rows, dim=-1)
        s_prob = torch.softmax(s_rows, dim=-1)
        return kl_dist(t_prob, s_prob)

    def total_distill_loss(self, keys: List[str], cfg: Optional[DistillationConfig] = None):
        config = cfg or self.distill_config
        config.validate()
        if not config.enabled:
            device, dtype = self._infer_device_dtype()
            return self._zero(device=device, dtype=dtype)
        students = [str(d) for d in config.student_domains if str(d) and str(d) != str(config.teacher_domain)]
        if not students:
            device, dtype = self._infer_device_dtype()
            return self._zero(device=device, dtype=dtype)
        total = None
        for sd in students:
            emb = self.embedding_distill(
                str(config.teacher_domain),
                sd,
                keys,
                w_cos=float(config.embedding_weight),
                w_mse=float(config.mse_weight),
            )
            nbr = float(config.neighbor_kl_weight) * self.neighbor_kl_distill(
                str(config.teacher_domain),
                sd,
                keys,
                k=int(config.neighbor_k),
            )
            cms = float(config.cms_teacher_weight) * self.cms_teacher_distill(
                sd,
                keys,
                w_cos=float(config.embedding_weight),
                w_mse=float(config.mse_weight),
            )
            val = emb + nbr + cms
            total = val if total is None else total + val
        return total / float(max(1, len(students)))

