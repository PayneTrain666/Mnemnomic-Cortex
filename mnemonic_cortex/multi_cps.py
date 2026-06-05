from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCPSManager(nn.Module):
    """
    Domain registry + routing for multiple CPS instances.
    """

    def __init__(self):
        super().__init__()
        self.cps = nn.ModuleDict()
        self.fusers = nn.ModuleDict()

    def register(self, domain: str, cps, fuser):
        self.cps[domain] = cps
        self.fusers[domain] = fuser
        return self

    def has_domain(self, domain: str) -> bool:
        return domain in self.cps

    def get(self, domain: str):
        if domain not in self.cps:
            raise KeyError(f"Domain '{domain}' not registered. Available: {list(self.cps.keys())}")
        return self.cps[domain], self.fusers[domain]

    def route(self, key: str, fallback: str = "core") -> str:
        if ":" in key:
            dom, _ = key.split(":", 1)
            if dom in self.cps:
                return dom
        if fallback in self.cps:
            return fallback
        if not self.cps:
            raise ValueError("MultiCPSManager has no registered domains")
        return next(iter(self.cps.keys()))

    def ensure(
        self,
        key: str,
        domain: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        d = domain or self.route(key)
        return self.cps[d].ensure(key, device=device, dtype=dtype)

    def make_param_groups(self, lr_base=2e-3, lr_phase=8e-4):
        groups = []
        for cps in self.cps.values():
            groups.extend(cps.make_param_groups(lr_base=lr_base, lr_phase=lr_phase))
        return groups

    def cohesion_regularizer(self, cms, keys: List[str], weight: float = 1e-3):
        if not keys:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = 0.0
        count = 0
        for k in keys:
            d = self.route(k)
            cps, fuser = self.get(d)
            if k not in cps._registry or k not in cms.keys():
                continue
            fused_cps, _ = fuser.fuse(cps.get(k).view())
            fused_cms = cms.read(k)
            total = total + F.mse_loss(fused_cps, fused_cms)
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return weight * (total / count)

