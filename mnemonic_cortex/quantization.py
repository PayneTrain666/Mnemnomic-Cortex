"""
Plain-language summary
----------------------
What this file is for: Int8-style quantization helpers for CPS values.
How it fits in the system: Compresses or discretizes parameters for efficiency experiments.
Status: OPT-IN
Important notes for non-coders: Some related tests have failed in recent full suites — triage if enabling.
"""

from typing import Dict, Optional, Tuple

import torch


def wrap_angle(a: torch.Tensor):
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def quantize_int8_sym(x: torch.Tensor, group_size: Optional[int] = None):
    if group_size is None:
        s = x.abs().max().clamp_min(1e-8) / 127.0
        q = torch.clamp((x / s).round(), -127, 127).to(torch.int8)
        return q, s
    d = x.numel()
    if d % group_size != 0:
        return quantize_int8_sym(x, group_size=None)
    xf = x.view(-1, group_size)
    s = xf.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8) / 127.0
    q = torch.clamp((xf / s).round(), -127, 127).to(torch.int8)
    return q.view_as(x), s


def dequant_int8_sym(q: torch.Tensor, s):
    if isinstance(s, torch.Tensor) and s.dim() == 2:
        group_size = s.size(1)
        qf = q.view(-1, group_size).float() * s
        return qf.view_as(q.float())
    return q.float() * s


class QuantPolicy:
    def __init__(
        self,
        euclid_group: Optional[int] = 64,
        hyp_group: Optional[int] = 32,
        spher_group: Optional[int] = 32,
        fisher_mu_group: Optional[int] = 32,
        fisher_lv_group: Optional[int] = 32,
        phase_amp_group: Optional[int] = 32,
    ):
        self.euclid_group = euclid_group
        self.hyp_group = hyp_group
        self.spher_group = spher_group
        self.fisher_mu_group = fisher_mu_group
        self.fisher_lv_group = fisher_lv_group
        self.phase_amp_group = phase_amp_group


class CPSQuantizer:
    def __init__(self, policy: QuantPolicy = QuantPolicy()):
        self.policy = policy

    @torch.no_grad()
    def quantize_entry(self, up) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if hasattr(up, "euclid"):
            out["E"] = quantize_int8_sym(up.euclid.detach(), self.policy.euclid_group)
        if hasattr(up, "hyp"):
            out["H"] = quantize_int8_sym(up.hyp.detach(), self.policy.hyp_group)
        if hasattr(up, "spher"):
            out["S"] = quantize_int8_sym(up.spher.detach(), self.policy.spher_group)
        if hasattr(up, "fisher_mu"):
            out["F_mu"] = quantize_int8_sym(up.fisher_mu.detach(), self.policy.fisher_mu_group)
        if hasattr(up, "fisher_lv"):
            out["F_lv"] = quantize_int8_sym(up.fisher_lv.detach(), self.policy.fisher_lv_group)
        if hasattr(up, "torus"):
            out["T"] = quantize_int8_sym(up.torus.detach(), None)
        if hasattr(up, "phase_amp"):
            out["P_amp"] = quantize_int8_sym(up.phase_amp.detach(), self.policy.phase_amp_group)
        if hasattr(up, "phase_phi"):
            out["P_phi"] = quantize_int8_sym(up.phase_phi.detach(), None)
        return out

    @torch.no_grad()
    def dequant_view(self, qpack: Dict[str, Tuple[torch.Tensor, torch.Tensor]], device=None):
        device = device or torch.device("cpu")
        view = {}
        if "E" in qpack:
            q, s = qpack["E"]
            view["E"] = dequant_int8_sym(q.to(device), s.to(device))
        if "H" in qpack:
            q, s = qpack["H"]
            view["H"] = dequant_int8_sym(q.to(device), s.to(device))
        if "S" in qpack:
            q, s = qpack["S"]
            view["S"] = dequant_int8_sym(q.to(device), s.to(device))
        if "F_mu" in qpack or "F_lv" in qpack:
            mu = None
            lv = None
            if "F_mu" in qpack:
                q, s = qpack["F_mu"]
                mu = dequant_int8_sym(q.to(device), s.to(device))
            if "F_lv" in qpack:
                q, s = qpack["F_lv"]
                lv = dequant_int8_sym(q.to(device), s.to(device))
            view["F"] = (mu, lv)
        if "T" in qpack:
            q, s = qpack["T"]
            view["T"] = wrap_angle(dequant_int8_sym(q.to(device), s.to(device)))
        if "P_amp" in qpack or "P_phi" in qpack:
            amp = None
            phi = None
            if "P_amp" in qpack:
                q, s = qpack["P_amp"]
                amp = dequant_int8_sym(q.to(device), s.to(device)).clamp_min(0.0)
            if "P_phi" in qpack:
                q, s = qpack["P_phi"]
                phi = wrap_angle(dequant_int8_sym(q.to(device), s.to(device)))
            view["P"] = (amp, phi)
        return view

