from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedDomainRouter(nn.Module):
    def __init__(self, domain_list: List[str], d_in: int, hidden: int = 128, temp: float = 1.0):
        super().__init__()
        if len(domain_list) < 1:
            raise ValueError("domain_list must not be empty")
        self.domains = list(domain_list)
        self.temp = nn.Parameter(torch.tensor(float(temp)))
        self.rules: Dict[str, str] = {}
        self.priors = nn.Parameter(torch.zeros(len(domain_list)))
        self.gate = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(domain_list)),
        )
        self.calib = nn.Parameter(torch.tensor(0.0))

    def set_rule(self, prefix: str, domain: str):
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}'")
        self.rules[prefix] = domain

    def _rule_route(self, key: Optional[str]) -> Optional[int]:
        if key is None:
            return None
        for pref, dom in self.rules.items():
            if key.startswith(pref):
                return self.domains.index(dom)
        if ":" in key:
            dom, _ = key.split(":", 1)
            if dom in self.domains:
                return self.domains.index(dom)
        return None

    def forward(self, query_vec: torch.Tensor, key: Optional[str] = None, top_k: int = 2):
        single = query_vec.dim() == 1
        x = query_vec.unsqueeze(0) if single else query_vec
        logits = self.gate(x) + self.priors + self.calib
        probs = F.softmax(logits / self.temp.clamp_min(1e-3), dim=-1)
        if key is not None:
            ridx = self._rule_route(key)
            if ridx is not None:
                mask = torch.zeros_like(probs)
                mask[..., ridx] = 1.0
                probs = 0.9 * mask + 0.1 * probs
        k = min(int(top_k), probs.size(-1))
        topv, topi = torch.topk(probs, k=k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        if single:
            return topi[0].tolist(), topv[0].tolist(), probs[0]
        return topi, topv, probs

