import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_sphere(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _retract_hyp(z: torch.Tensor, max_r: float = 0.999) -> torch.Tensor:
    r = z.norm(dim=-1, keepdim=True) + 1e-9
    scale = torch.clamp(max_r / r, max=1.0)
    return z * scale


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class ConsolidatedMemoryCfg:
    d_model: int = 512
    d_hyp: int = 64
    d_spher: int = 64
    d_fisher: int = 64
    d_phase: int = 32
    use_euclid: bool = True
    use_hyp: bool = True
    use_spher: bool = True
    use_fisher: bool = True
    use_torus: bool = True
    use_phase: bool = True


class ConsolidatedMemoryUnit(nn.Module):
    def __init__(self, cfg: ConsolidatedMemoryCfg):
        super().__init__()
        self.cfg = cfg
        if cfg.use_euclid:
            self.E = nn.Parameter(torch.zeros(cfg.d_model))
        if cfg.use_hyp:
            self.H = nn.Parameter(torch.zeros(cfg.d_hyp))
        if cfg.use_spher:
            self.S = nn.Parameter(_project_sphere(torch.randn(cfg.d_spher)))
        if cfg.use_fisher:
            self.F_mu = nn.Parameter(torch.zeros(cfg.d_fisher))
            self.F_lv = nn.Parameter(torch.full((cfg.d_fisher,), -2.0))
        if cfg.use_torus:
            self.T = nn.Parameter(torch.zeros(2))
        if cfg.use_phase:
            self.P_amp = nn.Parameter(torch.ones(cfg.d_phase) * 0.5)
            self.P_phi = nn.Parameter(torch.zeros(cfg.d_phase))

        self.register_buffer("confidence", torch.tensor(0.1))
        self.register_buffer("access", torch.tensor(0))
        self.register_buffer("created_ts", torch.tensor(time.time()))
        self._provenance: List[Dict] = []
        self.domain: str = "general"
        self.tags: List[str] = []

    def view(self) -> Dict[str, torch.Tensor]:
        out = {}
        if self.cfg.use_euclid:
            out["E"] = self.E
        if self.cfg.use_hyp:
            out["H"] = self.H
        if self.cfg.use_spher:
            out["S"] = self.S
        if self.cfg.use_fisher:
            out["F"] = (self.F_mu, self.F_lv)
        if self.cfg.use_torus:
            out["T"] = self.T
        if self.cfg.use_phase:
            out["P"] = (self.P_amp, self.P_phi)
        return out

    @torch.no_grad()
    def project_after_update(self):
        if self.cfg.use_hyp:
            self.H.copy_(_retract_hyp(self.H))
        if self.cfg.use_spher:
            self.S.copy_(_project_sphere(self.S))
        if self.cfg.use_torus:
            self.T.copy_(_wrap_angle(self.T))
        if self.cfg.use_phase:
            self.P_amp.copy_(torch.clamp(self.P_amp, 0.0, 10.0))
            self.P_phi.copy_(_wrap_angle(self.P_phi))

    @torch.no_grad()
    def ema_merge(self, candidate: Dict[str, torch.Tensor], alpha: float, src_info: Optional[Dict] = None):
        alpha = float(min(max(alpha, 0.0), 1.0))
        cur = self.view()
        if "E" in cur and "E" in candidate:
            self.E.copy_((1 - alpha) * self.E + alpha * candidate["E"])
        if "H" in cur and "H" in candidate:
            self.H.copy_((1 - alpha) * self.H + alpha * candidate["H"])
        if "S" in cur and "S" in candidate:
            self.S.copy_((1 - alpha) * self.S + alpha * candidate["S"])
        if "F" in cur and "F" in candidate:
            mu, lv = cur["F"]
            cmu, clv = candidate["F"]
            self.F_mu.copy_((1 - alpha) * mu + alpha * cmu)
            self.F_lv.copy_(torch.logaddexp((1 - alpha) * lv, alpha * clv))
        if "T" in cur and "T" in candidate:
            self.T.copy_((1 - alpha) * self.T + alpha * candidate["T"])
        if "P" in cur and "P" in candidate:
            amp, phi = cur["P"]
            camp, cphi = candidate["P"]
            self.P_amp.copy_((1 - alpha) * amp + alpha * camp)
            dphi = (cphi - phi + math.pi) % (2 * math.pi) - math.pi
            self.P_phi.copy_(phi + alpha * dphi)
        self.project_after_update()
        self.confidence = torch.clamp(self.confidence + alpha * 0.1, 0.0, 1.0)
        self.access += 1
        if src_info is not None:
            self._provenance.append(src_info)


class ConsolidatedMemoryStore(nn.Module):
    def __init__(self, default_cfg: ConsolidatedMemoryCfg = ConsolidatedMemoryCfg()):
        super().__init__()
        self.default_cfg = default_cfg
        self._registry = nn.ModuleDict()
        self.read_norm = nn.LayerNorm(default_cfg.d_model)
        self.sim_temp = nn.Parameter(torch.tensor(1.0))
        self._creation_hooks = []

    def register_creation_hook(self, hook):
        self._creation_hooks.append(hook)
        return self

    @staticmethod
    def _summarize_tensor(t: torch.Tensor):
        flat = t.detach().reshape(-1).float()
        pv = flat[: min(8, flat.numel())].tolist()
        return {
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
            "preview": [float(x) for x in pv],
        }

    def ensure(self, key: str, cfg: Optional[ConsolidatedMemoryCfg] = None) -> ConsolidatedMemoryUnit:
        if key in self._registry:
            return self._registry[key]
        unit = ConsolidatedMemoryUnit(cfg or self.default_cfg)
        self._registry[key] = unit
        if self._creation_hooks:
            payload = {
                "store": "ConsolidatedMemoryStore",
                "key": str(key),
                "params": {n: self._summarize_tensor(p) for n, p in unit.named_parameters(recurse=False)},
            }
            for hook in self._creation_hooks:
                try:
                    hook(payload)
                except Exception:
                    pass
        return unit

    def get(self, key: str) -> ConsolidatedMemoryUnit:
        return self._registry[key]

    def keys(self) -> List[str]:
        return list(self._registry.keys())

    @torch.no_grad()
    def write(self, key: str, candidate_view: Dict[str, torch.Tensor], alpha: float, src_info: Optional[Dict] = None):
        self.ensure(key).ema_merge(candidate_view, alpha=alpha, src_info=src_info)

    def _fused(self, unit: ConsolidatedMemoryUnit) -> torch.Tensor:
        if unit.cfg.use_euclid:
            return self.read_norm(unit.E)
        return torch.zeros(self.default_cfg.d_model, device=next(self.parameters()).device)

    @torch.no_grad()
    def read(self, key: str) -> torch.Tensor:
        return self._fused(self.get(key))

    @torch.no_grad()
    def knn(self, query_vec: torch.Tensor, k: int = 8) -> List[Tuple[str, float]]:
        if len(self._registry) == 0:
            return []
        q = F.normalize(query_vec.to(next(self.parameters()).device), dim=-1)
        keys = self.keys()
        mat = torch.stack([F.normalize(self._fused(self._registry[kk]), dim=-1) for kk in keys], dim=0)
        sim = (mat @ q) * self.sim_temp
        topv, topi = torch.topk(sim, k=min(k, len(keys)))
        return [(keys[i.item()], float(topv[j].item())) for j, i in enumerate(topi)]

    def extra_state_dict(self) -> Dict:
        meta = {}
        for k, u in self._registry.items():
            meta[k] = {
                "confidence": float(u.confidence.item()),
                "access": int(u.access.item()),
                "created_ts": float(u.created_ts.item()),
                "domain": getattr(u, "domain", "general"),
                "tags": list(getattr(u, "tags", [])),
            }
        return {"meta": meta, "version": 1}

    @torch.no_grad()
    def load_extra_state_dict(self, state: Dict):
        for k, m in state.get("meta", {}).items():
            if k not in self._registry:
                continue
            u = self._registry[k]
            device = next(u.parameters()).device
            u.confidence = torch.tensor(float(m.get("confidence", 0.1)), device=device)
            u.access = torch.tensor(int(m.get("access", 0)), device=device)
            u.created_ts = torch.tensor(float(m.get("created_ts", time.time())), device=device)
            u.domain = str(m.get("domain", "general"))
            u.tags = [str(t) for t in m.get("tags", [])]

    @torch.no_grad()
    def filter_keys(self, domain: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        ks = self.keys()
        if domain is not None:
            ks = [k for k in ks if getattr(self._registry[k], "domain", "general") == domain]
        if tag is not None:
            ks = [k for k in ks if tag in getattr(self._registry[k], "tags", [])]
        return ks

