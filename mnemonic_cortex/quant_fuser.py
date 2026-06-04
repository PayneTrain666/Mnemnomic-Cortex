from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _polar_to_cart(amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.cat([amp * torch.cos(phi), amp * torch.sin(phi)], dim=-1)


def _angles_to_sincos(theta: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)


class QuantAwareCPSFuser(nn.Module):
    def __init__(self, d_out: int, dims: Dict[str, int], use_heads=("E", "H", "S", "F", "T", "P"), quantizer=None):
        super().__init__()
        self.d_out = int(d_out)
        self.use_heads = tuple(use_heads)
        self.quantizer = quantizer
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

        self.proj_E = nn.Linear(dims.get("E", 0), self.d_out, bias=False) if "E" in self.use_heads and dims.get("E", 0) > 0 else None
        self.proj_H = nn.Linear(dims.get("H", 0), self.d_out, bias=False) if "H" in self.use_heads and dims.get("H", 0) > 0 else None
        self.proj_S = nn.Linear(dims.get("S", 0), self.d_out, bias=False) if "S" in self.use_heads and dims.get("S", 0) > 0 else None
        self.proj_F_mu = nn.Linear(dims.get("F", 0), self.d_out, bias=False) if "F" in self.use_heads and dims.get("F", 0) > 0 else None
        self.proj_F_lv = nn.Linear(dims.get("F", 0), self.d_out, bias=False) if "F" in self.use_heads and dims.get("F", 0) > 0 else None
        self.proj_T = nn.Linear(2 * dims.get("T", 0), self.d_out, bias=False) if "T" in self.use_heads and dims.get("T", 0) > 0 else None
        self.proj_P = nn.Linear(2 * dims.get("P", 0), self.d_out, bias=False) if "P" in self.use_heads and dims.get("P", 0) > 0 else None
        self.gates = nn.ParameterDict()
        if self.proj_E is not None:
            self.gates["E"] = nn.Parameter(torch.zeros(1))
        if self.proj_H is not None:
            self.gates["H"] = nn.Parameter(torch.zeros(1))
        if self.proj_S is not None:
            self.gates["S"] = nn.Parameter(torch.zeros(1))
        if self.proj_F_mu is not None:
            self.gates["F"] = nn.Parameter(torch.zeros(1))
        if self.proj_T is not None:
            self.gates["T"] = nn.Parameter(torch.zeros(1))
        if self.proj_P is not None:
            self.gates["P"] = nn.Parameter(torch.zeros(1))

    def _view_from_unit(self, unit):
        return unit.view() if hasattr(unit, "view") else {}

    @torch.no_grad()
    def _to_view(self, unit=None, qpack: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None, device=None):
        if unit is not None:
            v = self._view_from_unit(unit)
            for k, val in list(v.items()):
                if isinstance(val, tuple):
                    v[k] = tuple(t.to(device) if t is not None else None for t in val)
                else:
                    v[k] = val.to(device)
            return v
        if qpack is not None and self.quantizer is not None:
            return self.quantizer.dequant_view(qpack, device=device)
        raise ValueError("Need either `unit` or (`qpack` and `quantizer`).")

    def forward(self, unit=None, qpack: Optional[Dict] = None, device=None):
        device = device or next(self.parameters()).device
        view = self._to_view(unit=unit, qpack=qpack, device=device)
        contribs = []
        names = []

        if self.proj_E is not None and "E" in view:
            contribs.append(self.proj_E(F.normalize(view["E"].float(), dim=-1)))
            names.append("E")
        if self.proj_H is not None and "H" in view:
            contribs.append(self.proj_H(view["H"].float()))
            names.append("H")
        if self.proj_S is not None and "S" in view:
            contribs.append(self.proj_S(F.normalize(view["S"].float(), dim=-1)))
            names.append("S")
        if self.proj_F_mu is not None and "F" in view:
            mu, lv = view["F"]
            f = 0.0
            if mu is not None:
                f = f + self.proj_F_mu(mu.float())
            if lv is not None:
                f = f + self.proj_F_lv(lv.float())
            if not isinstance(f, float):
                contribs.append(f)
                names.append("F")
        if self.proj_T is not None and "T" in view and view["T"] is not None:
            contribs.append(self.proj_T(_angles_to_sincos(view["T"].float())))
            names.append("T")
        if self.proj_P is not None and "P" in view:
            amp, phi = view["P"]
            if amp is not None and phi is not None:
                contribs.append(self.proj_P(_polar_to_cart(amp.float(), phi.float())))
                names.append("P")

        if not contribs:
            return torch.zeros(self.d_out, device=device), {"weights": {}, "per_head": {}}

        c = torch.stack(contribs, dim=0)
        gate_logits = torch.stack([self.gates[nm] for nm in names], dim=0).view(-1)
        temp = self.gate_temp
        if not torch.isfinite(temp):
            temp = torch.tensor(1.0, device=gate_logits.device, dtype=gate_logits.dtype)
        w = F.softmax(gate_logits / temp.clamp_min(1e-3), dim=0)
        if not torch.isfinite(w).all():
            w = torch.full_like(w, 1.0 / max(1, w.numel()))
        fused = (w.unsqueeze(-1) * c).sum(dim=0)
        aux = {
            "weights": {n: float(w[i].detach().item()) for i, n in enumerate(names)},
            "per_head": {n: c[i].detach() for i, n in enumerate(names)},
        }
        return fused, aux

