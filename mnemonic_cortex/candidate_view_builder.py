"""
Plain-language summary
----------------------
What this file is for: Builds candidate memory 'views' for comparison or selection.
How it fits in the system: Turns memory contents into comparable candidate packages.
Status: ACTIVE / WORKING
Important notes for non-coders: Used when the system must pick among memory candidates.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryToViewAdapter(nn.Module):
    """
    Converts subsystem latent vectors into a normalized candidate view dict.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        d_hyp: int = 64,
        d_spher: int = 64,
        d_fisher: int = 64,
        d_phase: int = 32,
        d_torus: int = 2,
    ):
        super().__init__()
        self.map_E = nn.Linear(d_in, d_model)
        self.map_H = nn.Linear(d_in, d_hyp)
        self.map_S = nn.Linear(d_in, d_spher)
        self.map_F_mu = nn.Linear(d_in, d_fisher)
        self.map_F_lv = nn.Linear(d_in, d_fisher)
        self.map_T = nn.Linear(d_in, d_torus)
        self.map_P_amp = nn.Linear(d_in, d_phase)
        self.map_P_phi = nn.Linear(d_in, d_phase)

    @staticmethod
    def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
        return (x + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(self, latent: torch.Tensor, use_heads=("E", "H", "S", "F", "T", "P")) -> Dict[str, torch.Tensor]:
        single = latent.dim() == 1
        x = latent.unsqueeze(0) if single else latent
        out: Dict[str, torch.Tensor] = {}

        if "E" in use_heads:
            out["E"] = self.map_E(x)
        if "H" in use_heads:
            out["H"] = self.map_H(x)
        if "S" in use_heads:
            out["S"] = F.normalize(self.map_S(x), dim=-1)
        if "F" in use_heads:
            mu = self.map_F_mu(x)
            lv = self.map_F_lv(x).clamp(min=-8.0, max=6.0)
            out["F"] = (mu, lv)
        if "T" in use_heads:
            out["T"] = self._wrap_angle(self.map_T(x))
        if "P" in use_heads:
            amp = F.relu(self.map_P_amp(x))
            phi = self._wrap_angle(self.map_P_phi(x))
            out["P"] = (amp, phi)

        if single:
            for k, v in list(out.items()):
                if isinstance(v, tuple):
                    out[k] = tuple(t.squeeze(0) for t in v)
                else:
                    out[k] = v.squeeze(0)
        return out


def build_candidate_view(adapter: MemoryToViewAdapter, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
    return adapter(latent)

