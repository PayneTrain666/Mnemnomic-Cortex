from typing import Dict, Optional, Tuple

import torch


def wrap_angle(a: torch.Tensor):
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def quantize_int8_sym(x: torch.Tensor, group_size: Optional[int] = None):
    if x.numel() == 0:
        z = torch.zeros_like(x, dtype=torch.int8)
        return z, torch.tensor(1.0, device=x.device, dtype=torch.float32)
    if group_size is None:
        s = x.abs().max().clamp_min(1e-8) / 127.0
        q = torch.clamp((x / s).round(), -127, 127).to(torch.int8)
        return q, s
    d = x.numel()
    if int(group_size) <= 0 or d % int(group_size) != 0:
        return quantize_int8_sym(x, group_size=None)
    xf = x.reshape(-1, int(group_size))
    s = xf.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8) / 127.0
    q = torch.clamp((xf / s).round(), -127, 127).to(torch.int8)
    # Return per-group scales as [num_groups] for simpler dequant reconstruction.
    return q.view_as(x), s.view(-1)


def dequant_int8_sym(q: torch.Tensor, s):
    qf = q.float()
    if not isinstance(s, torch.Tensor):
        return qf * float(s)
    if s.dim() == 0:
        return qf * s.to(device=q.device, dtype=qf.dtype)
    if s.dim() == 1:
        num_groups = int(s.numel())
        if num_groups <= 0:
            return qf
        if q.numel() % num_groups != 0:
            # Fallback when caller supplies mismatched grouped scales.
            return qf * s.mean().to(device=q.device, dtype=qf.dtype)
        group_size = q.numel() // num_groups
        qg = qf.reshape(num_groups, group_size)
        sg = s.to(device=q.device, dtype=qf.dtype).view(num_groups, 1)
        return (qg * sg).reshape_as(qf)
    if s.dim() == 2:
        num_groups = int(s.size(0))
        if num_groups <= 0 or q.numel() % num_groups != 0:
            return qf * s.mean().to(device=q.device, dtype=qf.dtype)
        group_size = q.numel() // num_groups
        qg = qf.reshape(num_groups, group_size)
        sg = s.to(device=q.device, dtype=qf.dtype)
        if sg.size(1) == 1:
            return (qg * sg).reshape_as(qf)
        if sg.size(1) == group_size:
            return (qg * sg).reshape_as(qf)
        return qf * sg.mean()
    return qf * s.mean().to(device=q.device, dtype=qf.dtype)


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
    def __init__(self, policy: Optional[QuantPolicy] = None):
        self.policy = policy if policy is not None else QuantPolicy()

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
            view["E"] = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s)
        if "H" in qpack:
            q, s = qpack["H"]
            view["H"] = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s)
        if "S" in qpack:
            q, s = qpack["S"]
            view["S"] = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s)
        if "F_mu" in qpack or "F_lv" in qpack:
            mu = None
            lv = None
            if "F_mu" in qpack:
                q, s = qpack["F_mu"]
                mu = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s)
            if "F_lv" in qpack:
                q, s = qpack["F_lv"]
                lv = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s)
            view["F"] = (mu, lv)
        if "T" in qpack:
            q, s = qpack["T"]
            view["T"] = wrap_angle(dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s))
        if "P_amp" in qpack or "P_phi" in qpack:
            amp = None
            phi = None
            if "P_amp" in qpack:
                q, s = qpack["P_amp"]
                amp = dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s).clamp_min(0.0)
            if "P_phi" in qpack:
                q, s = qpack["P_phi"]
                phi = wrap_angle(dequant_int8_sym(q.to(device), s.to(device) if isinstance(s, torch.Tensor) else s))
            view["P"] = (amp, phi)
        return view

