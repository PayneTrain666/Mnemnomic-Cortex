"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context depth adapter.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextDepthAdapter(nn.Module):
    """Adapt context triplet summary to each WM depth.

    Inputs:
    - context_triplet: [B,C,3,D]
    - depth_weights: [Z]
    - triplet_bias: [3]

    Output:
    - depth_context: [B,Z,1,3,D]
    """

    def __init__(self, dim: int, num_depths: int):
        super().__init__()
        self.dim = dim
        self.num_depths = num_depths
        self.depth_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(num_depths)
        ])

    def forward(self, context_triplet: torch.Tensor, depth_weights: torch.Tensor, triplet_bias: torch.Tensor) -> torch.Tensor:
        if context_triplet.dim() != 4 or context_triplet.size(-2) != 3:
            raise ValueError("Expected context_triplet [B,C,3,D]")
        b, c, three, d = context_triplet.shape
        summary = context_triplet.mean(dim=1)  # [B,3,D]
        out = []
        for z, adapter in enumerate(self.depth_adapters):
            adapted = adapter(summary)
            adapted = adapted * depth_weights[z].view(1, 1, 1) * triplet_bias.view(1, 3, 1)
            out.append(adapted)
        return torch.stack(out, dim=1).unsqueeze(2)  # [B,Z,1,3,D]
