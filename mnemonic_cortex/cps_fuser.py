from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FuserCfg:
    d_model: int = 512
    d_hyp: int = 64
    d_spher: int = 64
    d_fisher: int = 64
    d_phase: int = 32
    use_heads: Tuple[str, ...] = ("H", "S", "F", "P")
    alpha_heads: float = 0.25
    agree_weight: float = 1e-3


class CPSFuser(nn.Module):
    def __init__(self, cfg: FuserCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.map_H = nn.Linear(cfg.d_hyp, d)
        self.map_S = nn.Linear(cfg.d_spher, d)
        self.map_F = nn.Linear(cfg.d_fisher, d)
        self.map_P = nn.Linear(cfg.d_phase, d)
        self.gate_phase = nn.Sequential(nn.Linear(cfg.d_phase, d), nn.Sigmoid())

    def fuse(self, view: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "E" in view, "Euclidean head required for base."
        base = view["E"]
        pieces: List[torch.Tensor] = [base]
        agree_targets: List[torch.Tensor] = []

        if "H" in self.cfg.use_heads and "H" in view:
            h = self.map_H(view["H"])
            pieces.append(self.cfg.alpha_heads * h)
            agree_targets.append(h)
        if "S" in self.cfg.use_heads and "S" in view:
            s = self.map_S(view["S"])
            pieces.append(self.cfg.alpha_heads * s)
            agree_targets.append(s)
        if "F" in self.cfg.use_heads and "F" in view:
            mu, _lv = view["F"]
            f = self.map_F(mu)
            pieces.append(self.cfg.alpha_heads * f)
            agree_targets.append(f)
        if "P" in self.cfg.use_heads and "P" in view:
            amp, phi = view["P"]
            gate = self.gate_phase(phi)
            p = self.map_P(amp) * gate
            pieces.append(self.cfg.alpha_heads * p)
            agree_targets.append(self.map_P(amp))

        fused = torch.stack(pieces, dim=0).sum(dim=0)
        if agree_targets:
            tgt = torch.stack(agree_targets, dim=0).mean(dim=0).detach()
            agree = F.mse_loss(base, tgt)
        else:
            agree = base.new_tensor(0.0)
        return fused, self.cfg.agree_weight * agree

