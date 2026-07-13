"""
Plain-language summary
----------------------
What this file is for: Orchestrates consolidation across memory banks.
How it fits in the system: Coordinates when short-term patterns become longer-term stores.
Status: ACTIVE when consolidation path is on
Important notes for non-coders: Works with brokers and CMS helpers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryConsolidationManagerV2(nn.Module):
    """Topology prototype consolidator/recaller for LTM fusion."""

    def __init__(self, embedding_dim: int, topological_dim: int = 128):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.topological_dim = int(topological_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.embedding_dim * 3 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, self.topological_dim),
        )
        self.memory = nn.Parameter(
            torch.randn(32, self.topological_dim) * 0.02
        )
        self.usage = nn.Parameter(torch.zeros(32), requires_grad=False)
        self.query_proj = nn.Linear(self.embedding_dim, self.topological_dim)
        self.read_attn = nn.MultiheadAttention(
            self.topological_dim, num_heads=self._pick_heads(self.topological_dim), batch_first=True
        )

    @staticmethod
    def _pick_heads(dim: int) -> int:
        for h in (8, 4, 2, 1):
            if dim % h == 0:
                return h
        return 1

    def consolidate(self, pooled: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        # pooled: [B,3,D] or [B,D] — mean over systems when needed
        if pooled.dim() == 3:
            pooled = pooled.mean(dim=1)
        imp = importance
        if imp.dim() == 3:
            imp = imp.mean(dim=1)
        if imp.dim() == 2 and imp.size(-1) != 1:
            imp = imp.mean(dim=-1, keepdim=True)
        feat = torch.cat([pooled, pooled, pooled, imp], dim=-1)
        proto = self.encode(feat)
        with torch.no_grad():
            slot = int(torch.argmin(self.usage).item())
            alpha = float(imp.mean().clamp(0.05, 0.95).item())
            self.memory[slot] = (1.0 - alpha) * self.memory[slot] + alpha * proto.mean(dim=0)
            self.usage[slot] = self.usage[slot] + alpha
        return proto

    def recall(self, query: torch.Tensor, topk: int = 1) -> torch.Tensor:
        # query: [B,T,D] or [B,D]
        if query.dim() == 3:
            q = query.mean(dim=1)
        else:
            q = query
        bank = self.memory.detach().clone().unsqueeze(0).expand(q.size(0), -1, -1)
        q_tok = self.query_proj(q).unsqueeze(1)
        recalled, _ = self.read_attn(q_tok, bank, bank, need_weights=False)
        k = min(int(topk), bank.size(1))
        return recalled[:, :k, :]
