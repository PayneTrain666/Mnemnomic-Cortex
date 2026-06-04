from typing import List, Optional

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
        domain = str(domain).strip()
        if not domain:
            raise ValueError("Domain name must be a non-empty string")
        self.cps[domain] = cps
        self.fusers[domain] = fuser
        return self

    def domains(self) -> List[str]:
        return list(self.cps.keys())

    def has_domain(self, domain: str) -> bool:
        return domain in self.cps

    def get(self, domain: str):
        if domain not in self.cps:
            raise KeyError(f"Domain '{domain}' not registered. Available: {self.domains()}")
        return self.cps[domain], self.fusers[domain]

    @staticmethod
    def _parse_domain_from_key(key: str) -> Optional[str]:
        if not isinstance(key, str):
            return None
        if ":" not in key:
            return None
        dom, _ = key.split(":", 1)
        dom = dom.strip()
        return dom if dom else None

    def route(self, key: str, fallback: str = "core") -> str:
        dom = self._parse_domain_from_key(key)
        if dom is not None and dom in self.cps:
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
        if d not in self.cps:
            raise KeyError(f"Domain '{d}' not registered. Available: {self.domains()}")
        return self.cps[d].ensure(key, device=device, dtype=dtype)

    def make_param_groups(self, lr_base=2e-3, lr_phase=8e-4):
        groups = []
        for cps in self.cps.values():
            groups.extend(cps.make_param_groups(lr_base=lr_base, lr_phase=lr_phase))
        return groups

    def _device_or_cpu(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def cohesion_regularizer(self, cms, keys: List[str], weight: float = 1e-3):
        if not keys:
            return torch.tensor(0.0, device=self._device_or_cpu())
        total = torch.tensor(0.0, device=self._device_or_cpu())
        count = 0
        cms_keys = set(cms.keys())
        for k in keys:
            d = self.route(k)
            cps, fuser = self.get(d)
            if k not in cps._registry or k not in cms_keys:
                continue
            fused_cps, _ = fuser.fuse(cps.get(k).view())
            fused_cms = cms.read(k)
            total = total + F.mse_loss(fused_cps, fused_cms)
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=self._device_or_cpu())
        return weight * (total / count)

