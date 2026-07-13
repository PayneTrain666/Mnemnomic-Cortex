"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component:  wm light transformer.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LightweightTransformerBlock(nn.Module):
    """Small deterministic transformer-style block used to keep tests fast.

    It performs real multi-head self-attention + feed-forward processing, but
    avoids the heavier PyTorch TransformerEncoder wrapper.
    """

    def __init__(self, dim: int, num_heads: int = 4, ff_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_multiplier, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,L,D]
        n, l, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(n, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [N,L,H,HD]
        q = q.transpose(1, 2)  # [N,H,L,HD]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v).transpose(1, 2).contiguous().view(n, l, d)
        x = x + self.dropout(self.out(attended))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class LightweightTransformerStack(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 1, ff_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LightweightTransformerBlock(dim, num_heads, ff_multiplier, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
