"""
Plain-language summary
----------------------
What this file is for: Long-term memory package module: reasoning stack.
How it fits in the system: Supports LTM banks, MANN/geometry helpers, or package wiring used with cortex LTM.
Status: ACTIVE / LEGACY depending on file
Important notes for non-coders: Some files are local copies or aliases; prefer top-level cortex + triple_hybrid for product runtime.

Technical notes (original):
Optional global reasoning stack for memory readout tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_utils import TransformerStack


class ReasoningStack(nn.Module):
    def __init__(self, dim: int, depth: int = 1, heads: int = 4, dropout: float = 0.0, ffn_mult: int = 2, max_len: int = 128):
        super().__init__()
        self.stack = TransformerStack(dim, depth, heads, dropout, ffn_mult, max_len)

    def forward(self, tokens: torch.Tensor, mem_tokens: torch.Tensor | None = None, need_attn: bool = False):
        x = torch.cat([tokens, mem_tokens], dim=1) if mem_tokens is not None else tokens
        return self.stack(x, need_weights=need_attn)
