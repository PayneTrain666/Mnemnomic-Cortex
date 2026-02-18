from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryBlender(nn.Module):
    """
    Slot-wise geometry blend for curvature with extended mode awareness.

    Extended mode order:
      [hyperbolic, spherical, euclidean, fractal, torus, cp]
    """

    def __init__(self, num_slots: int, init_scale: float = 0.05):
        super().__init__()
        self.num_slots = int(num_slots)
        # Per-slot curvature seeds for each extended mode.
        self.curv_h = nn.Parameter(init_scale * torch.randn(self.num_slots))
        self.curv_s = nn.Parameter(init_scale * torch.randn(self.num_slots))
        self.curv_e = nn.Parameter(torch.zeros(self.num_slots))
        self.curv_f = nn.Parameter(init_scale * torch.randn(self.num_slots))
        self.curv_t = nn.Parameter(init_scale * torch.randn(self.num_slots))
        self.curv_cp = nn.Parameter(init_scale * torch.randn(self.num_slots))
        # Start close to uniform mode usage.
        self.mode_logits = nn.Parameter(torch.zeros(6))

    @staticmethod
    def _map_ext_to_legacy4(w_ext: torch.Tensor) -> torch.Tensor:
        # Keep backward compatibility with 4-mode metric heads.
        w_h = w_ext[0]
        w_s = w_ext[1]
        w_e = w_ext[2]
        w_f = w_ext[3] + 0.5 * w_ext[4] + 0.5 * w_ext[5]
        out = torch.stack([w_h, w_s, w_e, w_f], dim=0)
        out = out / out.sum().clamp_min(1e-8)
        return out

    @torch.no_grad()
    def set_mode_priors(self, priors: torch.Tensor, mix: float = 0.05):
        """
        Gently nudge learned mode logits toward priors.
        """
        p = priors.to(self.mode_logits.device, self.mode_logits.dtype).view(-1)
        if p.numel() != self.mode_logits.numel():
            return
        p = p / p.sum().clamp_min(1e-8)
        cur = torch.softmax(self.mode_logits, dim=-1)
        mix = float(max(0.0, min(1.0, mix)))
        nxt = (1.0 - mix) * cur + mix * p
        self.mode_logits.copy_(torch.log(nxt.clamp_min(1e-8)))

    def forward(
        self, query_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        _ = query_feat
        w_ext = F.softmax(self.mode_logits, dim=-1)  # (6,)
        curv = (
            w_ext[0] * self.curv_h
            + w_ext[1] * self.curv_s
            + w_ext[2] * self.curv_e
            + w_ext[3] * self.curv_f
            + w_ext[4] * self.curv_t
            + w_ext[5] * self.curv_cp
        ).clamp(-1.0, 1.0)
        w4 = self._map_ext_to_legacy4(w_ext)
        wext_dict = {
            "hyperbolic": float(w_ext[0].detach().item()),
            "spherical": float(w_ext[1].detach().item()),
            "euclidean": float(w_ext[2].detach().item()),
            "fractal": float(w_ext[3].detach().item()),
            "torus": float(w_ext[4].detach().item()),
            "cp": float(w_ext[5].detach().item()),
        }
        return curv, w4, wext_dict

