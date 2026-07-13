"""
Plain-language summary
----------------------
What this file is for: Consolidated Parameter Store: shared parameter / concept geometry store.
How it fits in the system: Holds unified parameters that multiple domains can share.
Status: ACTIVE / OPT-IN by feature
Important notes for non-coders: Often paired with CPS fuser and multi-CPS manager.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

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
            self.hyp.copy_(_retract_hyp(self.hyp))
        if self.cfg.use_spher:
            self.spher.copy_(_project_sphere(self.spher))
        if self.cfg.use_torus:
            self.torus.copy_(_wrap_angle(self.torus))
        if self.cfg.use_phase:
            self.phase_amp.copy_(torch.clamp(self.phase_amp, 0.0, 10.0))
            self.phase_phi.copy_(_wrap_angle(self.phase_phi))

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

    @staticmethod
    def _align_param_module(
        up: UnifiedParam,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> UnifiedParam:
        if device is None and dtype is None:
            return up
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        up.to(**kwargs)
        return up

    def ensure(
        self,
        key: str,
        cfg: UnifiedParamCfg = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> UnifiedParam:
        if key in self._registry:
            return self._align_param_module(self._registry[key], device=device, dtype=dtype)
        up = UnifiedParam(cfg or self.default_cfg)
        up = self._align_param_module(up, device=device, dtype=dtype)
        self._registry[key] = up
        if self._creation_hooks:
            payload = {
                "store": "ConsolidatedParamStore",
                "key": str(key),
                "params": {n: self._summarize_tensor(p) for n, p in up.named_parameters(recurse=False)},
            }
            for hook in self._creation_hooks:
                try:
                    hook(payload)
                except Exception:
                    pass
        return up

    def get(self, key: str) -> UnifiedParam:
        return self._registry[key]

    def keys(self) -> List[str]:
        return list(self._registry.keys())

    def parameters_for_optim(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        for up in self._registry.values():
            out += up.subparams()
        return out

    def make_param_groups(self, lr_base: float = 2e-3, lr_phase: float = 8e-4):
        groups = []
        for up in self._registry.values():
            for name, p in up.named_parameters(recurse=False):
                if "phase" in name:
                    groups.append({"params": [p], "lr": lr_phase})
                else:
                    groups.append({"params": [p], "lr": lr_base})
        return groups

    def freeze(self, keys: List[str], heads: Optional[List[str]] = None):
        wanted = None if heads is None else {h.upper() for h in heads}
        for k in keys:
            up = self._registry.get(k, None)
            if up is None:
                continue
            for name, p in up.named_parameters(recurse=False):
                code = "E"
                if name.startswith("hyp"):
                    code = "H"
                elif name.startswith("spher"):
                    code = "S"
                elif name.startswith("fisher"):
                    code = "F"
                elif name.startswith("torus"):
                    code = "T"
                elif name.startswith("phase"):
                    code = "P"
                if wanted is None or code in wanted:
                    p.requires_grad_(False)

    def unfreeze(self, keys: List[str], heads: Optional[List[str]] = None):
        wanted = None if heads is None else {h.upper() for h in heads}
        for k in keys:
            up = self._registry.get(k, None)
            if up is None:
                continue
            for name, p in up.named_parameters(recurse=False):
                code = "E"
                if name.startswith("hyp"):
                    code = "H"
                elif name.startswith("spher"):
                    code = "S"
                elif name.startswith("fisher"):
                    code = "F"
                elif name.startswith("torus"):
                    code = "T"
                elif name.startswith("phase"):
                    code = "P"
                if wanted is None or code in wanted:
                    p.requires_grad_(True)

    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        snap: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, up in self._registry.items():
            snap[k] = {n: p.detach().clone() for n, p in up.named_parameters(recurse=False)}
        return snap

    @torch.no_grad()
    def restore(self, snap: Dict[str, Dict[str, torch.Tensor]], strict: bool = False):
        for k, params in snap.items():
            if k not in self._registry:
                if strict:
                    raise KeyError(f"Missing key {k} in CPS")
                continue
            up = self._registry[k]
            for n, t in params.items():
                if hasattr(up, n):
                    target = getattr(up, n)
                    target.copy_(t.to(target.device))

    @torch.no_grad()
    def confidence_signals_for_keys(self, keys: List[str]) -> Dict[str, float]:
        """
        Summarize CPS uncertainty and phase consistency for AHG-style gating.
        """
        if not keys:
            return {"fisher_uncertainty": 0.0, "phase_agreement": 0.0}
        fisher_vals = []
        phase_vals = []
        for k in keys:
            if k not in self._registry:
                continue
            up = self._registry[k]
            if hasattr(up, "fisher_lv"):
                fisher_vals.append(torch.exp(up.fisher_lv.detach()).mean())
            if hasattr(up, "phase_phi"):
                # Lower circular dispersion => stronger phase agreement.
                phi = up.phase_phi.detach()
                c = torch.cos(phi).mean()
                s = torch.sin(phi).mean()
                phase_vals.append(torch.sqrt(c * c + s * s))
        if fisher_vals:
            fisher = float(torch.stack(fisher_vals).mean().item())
        else:
            fisher = 0.0
        if phase_vals:
            phase = float(torch.stack(phase_vals).mean().item())
        else:
            phase = 0.0
        return {"fisher_uncertainty": fisher, "phase_agreement": phase}

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

