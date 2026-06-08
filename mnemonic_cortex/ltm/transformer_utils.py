"""Small transformer stack used by WM/reasoning fusion."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, ffn_mult: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        a, w = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=need_weights)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, w if need_weights else None


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int = 1, heads: int = 4, dropout: float = 0.0, ffn_mult: int = 2, max_len: int = 64, use_pos_emb: bool = True):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.01) if use_pos_emb else None
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout, ffn_mult) for _ in range(depth)])

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        if self.use_pos_emb:
            x = x + self.pos[:, : x.size(1), :]
        attn: List[torch.Tensor] = []
        for block in self.blocks:
            x, w = block(x, need_weights=need_weights)
            if need_weights and w is not None:
                attn.append(w)
        return x, attn if need_weights else None
