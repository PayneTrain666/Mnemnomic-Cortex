"""
Plain-language summary
----------------------
What this file is for: Index structures for looking up consolidated memory entries.
How it fits in the system: Speeds or organizes CMS addressing.
Status: ACTIVE when CMS path on
Important notes for non-coders: Supporting structure, not the full store.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


class CMSIndex:
    """
    Torch fallback index for CMS vectors.
    """

    def __init__(self, cms, device=None):
        self.cms = cms
        self.device = device
        self.keys: List[str] = []
        self.mat = None

    @torch.no_grad()
    def rebuild_all(self):
        self.keys = self.cms.keys()
        if not self.keys:
            self.mat = None
            return
        if self.device is None:
            self.device = next(self.cms.parameters()).device
        vecs = [F.normalize(self.cms.read(k).to(self.device), dim=-1) for k in self.keys]
        self.mat = torch.stack(vecs, dim=0)

    @torch.no_grad()
    def add_or_update(self, keys: List[str]):
        _ = keys
        self.rebuild_all()

    @torch.no_grad()
    def search(self, q: torch.Tensor, k: int = 8) -> List[Tuple[str, float]]:
        if self.mat is None or len(self.keys) == 0:
            return []
        qn = F.normalize(q.to(self.device), dim=-1)
        sim = self.mat @ qn
        topv, topi = torch.topk(sim, k=min(k, len(self.keys)))
        return [(self.keys[i.item()], float(topv[j].item())) for j, i in enumerate(topi)]

