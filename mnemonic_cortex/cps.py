import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


def _project_sphere(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _retract_hyp(z: torch.Tensor, max_r: float = 0.999) -> torch.Tensor:
    r = z.norm(dim=-1, keepdim=True) + 1e-9
    scale = torch.clamp(max_r / r, max=1.0)
    return z * scale


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class UnifiedParamCfg:
    d_euclid: int = 512
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


class UnifiedParam(nn.Module):
    def __init__(self, cfg: UnifiedParamCfg):
        super().__init__()
        self.cfg = cfg
        if cfg.use_euclid:
            self.euclid = nn.Parameter(torch.randn(cfg.d_euclid) * 0.02)
        if cfg.use_hyp:
            self.hyp = nn.Parameter(torch.randn(cfg.d_hyp) * 0.02)
        if cfg.use_spher:
            s = torch.randn(cfg.d_spher)
            self.spher = nn.Parameter(_project_sphere(s))
        if cfg.use_fisher:
            self.fisher_mu = nn.Parameter(torch.zeros(cfg.d_fisher))
            self.fisher_lv = nn.Parameter(torch.full((cfg.d_fisher,), -2.0))
        if cfg.use_torus:
            self.torus = nn.Parameter(torch.zeros(2))
        if cfg.use_phase:
            self.phase_amp = nn.Parameter(torch.ones(cfg.d_phase) * 0.5)
            self.phase_phi = nn.Parameter(torch.zeros(cfg.d_phase))

    def subparams(self) -> List[nn.Parameter]:
        return [p for _, p in self.named_parameters(recurse=False)]

    @torch.no_grad()
    def project_after_step(self):
        if self.cfg.use_hyp:
            self.hyp.data = _retract_hyp(self.hyp.data)
        if self.cfg.use_spher:
            self.spher.data = _project_sphere(self.spher.data)
        if self.cfg.use_torus:
            self.torus.data = _wrap_angle(self.torus.data)
        if self.cfg.use_phase:
            self.phase_amp.data = torch.clamp(self.phase_amp.data, 0.0, 10.0)
            self.phase_phi.data = _wrap_angle(self.phase_phi.data)

    def view(self) -> Dict[str, torch.Tensor]:
        out = {}
        if self.cfg.use_euclid:
            out["E"] = self.euclid
        if self.cfg.use_hyp:
            out["H"] = self.hyp
        if self.cfg.use_spher:
            out["S"] = self.spher
        if self.cfg.use_fisher:
            out["F"] = (self.fisher_mu, self.fisher_lv)
        if self.cfg.use_torus:
            out["T"] = self.torus
        if self.cfg.use_phase:
            out["P"] = (self.phase_amp, self.phase_phi)
        return out


class ConsolidatedParamStore(nn.Module):
    def __init__(self, default_cfg: UnifiedParamCfg = UnifiedParamCfg()):
        super().__init__()
        self._registry = nn.ModuleDict()
        self.default_cfg = default_cfg

    def ensure(self, key: str, cfg: UnifiedParamCfg = None) -> UnifiedParam:
        if key in self._registry:
            return self._registry[key]
        up = UnifiedParam(cfg or self.default_cfg)
        self._registry[key] = up
        return up

    def get(self, key: str) -> UnifiedParam:
        return self._registry[key]

    def parameters_for_optim(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        for up in self._registry.values():
            out += up.subparams()
        return out

    @torch.no_grad()
    def project_all(self):
        for up in self._registry.values():
            up.project_after_step()


class PolyOptim(torch.optim.Adam):
    def __init__(self, cps: ConsolidatedParamStore, **adam_kwargs):
        super().__init__(cps.parameters_for_optim(), **adam_kwargs)
        self.cps = cps

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.cps.project_all()
        return loss

