"""
Geometry kernel hooks with safe fallbacks.

This module is intentionally lightweight: if CUDA custom kernels are unavailable,
it falls back to pure PyTorch ops.
"""

from __future__ import annotations

import torch


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: (..., N, D), y: (..., M, D) -> (..., N, M)
    return torch.cdist(x, y, p=2)


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xn = x / (x.norm(dim=-1, keepdim=True) + eps)
    yn = y / (y.norm(dim=-1, keepdim=True) + eps)
    return torch.matmul(xn, yn.transpose(-2, -1))

