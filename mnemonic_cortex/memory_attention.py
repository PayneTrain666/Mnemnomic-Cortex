"""
Plain-language summary
----------------------
What this file is for: Multi-scale attention used when reading or writing memory.
How it fits in the system: Shared attention building block for memory modules.
Status: ACTIVE
Important notes for non-coders: Helps the model focus on the most relevant memory pieces.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MultiScaleAttention(nn.Module):
    """Local + pooled global attention branches for cross-memory fusion."""

    def __init__(self, embed_dim: int, num_heads: int, scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = int(embed_dim)
        self.scales = tuple(int(s) for s in scales if int(s) > 0) or (1,)
        self.branches = nn.ModuleList(
            [nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True) for _ in self.scales]
        )
        self.branch_gate = nn.Parameter(torch.zeros(len(self.scales)))

    @staticmethod
    def _pool_seq(x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1 or x.size(1) < scale:
            return x
        bsz, seq, dim = x.shape
        usable = (seq // scale) * scale
        if usable < scale:
            return x.mean(dim=1, keepdim=True)
        return x[:, :usable, :].reshape(bsz, usable // scale, scale, dim).mean(dim=2)

    @staticmethod
    def _match_query_len(query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.size(1) == query.size(1):
            return ref
        if ref.size(1) == 1:
            return ref.expand(-1, query.size(1), -1)
        reps = max(1, query.size(1) // ref.size(1))
        expanded = ref.repeat_interleave(reps, dim=1)
        return expanded[:, : query.size(1), :]

    def forward(self, query, key, value, need_weights=False):
        outs = []
        for scale, attn in zip(self.scales, self.branches):
            k = self._match_query_len(query, self._pool_seq(key, scale))
            v = self._match_query_len(query, self._pool_seq(value, scale))
            out, _ = attn(query, k, v, need_weights=False)
            outs.append(out)
        mix = torch.softmax(self.branch_gate, dim=0)
        merged = sum(mix[i] * outs[i] for i in range(len(outs)))
        if need_weights:
            return merged, None
        return merged, None
