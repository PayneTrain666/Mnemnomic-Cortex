"""
Plain-language summary
----------------------
What this file is for: Working-memory (QDT-WM) component: context triplet projector.
How it fits in the system: Part of the active scratchpad stack that sits between sensory input and long-term memory.
Status: ACTIVE / OPT-IN depending on flags
Important notes for non-coders: See qdt_working_memory.py for the main assembly; this file is one piece of that stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextTripletProjector(nn.Module):
    """Project context tokens into anchor/direction/phase triplets.

    Input:
    - context: [B,C,D]

    Output:
    - triplet: [B,C,3,D]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.to_triplet = nn.Linear(dim, 3 * dim)
        self.from_triplet = nn.Linear(3 * dim, dim)

    def project(self, context: torch.Tensor) -> torch.Tensor:
        b, c, d = context.shape
        return self.to_triplet(context).view(b, c, 3, d)

    def fuse(self, triplet: torch.Tensor) -> torch.Tensor:
        b, c, three, d = triplet.shape
        if three != 3:
            raise ValueError("Context triplet dimension must be 3")
        return self.from_triplet(triplet.reshape(b, c, 3 * d))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.project(context)
