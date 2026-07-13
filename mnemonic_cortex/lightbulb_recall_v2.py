"""
Plain-language summary
----------------------
What this file is for: Newer lightbulb-style explosive recall controller.
How it fits in the system: Modernized recall boost path used by some cortex builds.
Status: ACTIVE when enabled (prefer over v1 where wired)
Important notes for non-coders: Works with event logging.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RecallEvent:
    triggered: bool
    spike_score: float
    expanded_k: int
    hops: int


class LightbulbRecallV2(nn.Module):
    """
    Spike detector + bounded explosive recall controller.
    """

    def __init__(self, in_dim: int = 4, threshold: float = 0.75, max_hops: int = 2, k_expand_mult: float = 1.5):
        super().__init__()
        self.threshold = float(threshold)
        self.max_hops = int(max_hops)
        self.k_expand_mult = float(k_expand_mult)
        self.signal = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        entropy_drop: torch.Tensor,
        cms_cps_agreement: torch.Tensor,
        novelty: torch.Tensor,
        uncertainty: torch.Tensor,
        base_k: int,
        debounce_ok: bool = True,
    ) -> RecallEvent:
        x = torch.stack([entropy_drop, cms_cps_agreement, novelty, uncertainty], dim=-1).float()
        score = float(self.signal(x).mean().item())
        trig = bool(debounce_ok and score >= self.threshold)
        if trig:
            return RecallEvent(True, score, max(base_k, int(base_k * self.k_expand_mult)), self.max_hops)
        return RecallEvent(False, score, base_k, 0)
