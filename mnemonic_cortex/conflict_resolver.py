from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class ConflictCfg:
    cosine_neg_gate: float = -0.2
    min_strength: float = 0.35
    decay: float = 0.99


class ConflictResolver:
    def __init__(self, cms, cfg: ConflictCfg = ConflictCfg()):
        self.cms = cms
        self.cfg = cfg
        self._neg_counts: Dict[str, float] = {}

    @torch.no_grad()
    def assess(self, key: str, candidate_E: torch.Tensor) -> float:
        if key not in self.cms.keys():
            return 0.0
        cur = self.cms.read(key)
        return float(torch.dot(F.normalize(cur, dim=-1), F.normalize(candidate_E, dim=-1)).item())

    @torch.no_grad()
    def update_on_merge(self, key: str, candidate_E: torch.Tensor) -> bool:
        s = self.assess(key, candidate_E)
        self._neg_counts[key] = self.cfg.decay * self._neg_counts.get(key, 0.0)
        if s < self.cfg.cosine_neg_gate and candidate_E.norm().item() > self.cfg.min_strength:
            self._neg_counts[key] += 1.0
            return True
        return False

    def penalty(self, key: str) -> float:
        return min(0.3, 0.05 * self._neg_counts.get(key, 0.0))

