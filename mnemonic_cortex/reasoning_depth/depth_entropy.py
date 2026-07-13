"""
Plain-language summary
----------------------
What this file is for: Reasoning-depth component: depth entropy.
How it fits in the system: Supports multi-layer deeper routing across memory depths when enabled.
Status: OPT-IN
Important notes for non-coders: Many adapters stay off until a controller explicitly enables them.
"""

from __future__ import annotations
import torch
class DepthEntropyError(ValueError): pass
def validate_probability_tensor(probs: torch.Tensor, *, dim: int=-1, eps: float=1e-8) -> torch.Tensor:
    if not isinstance(probs, torch.Tensor): raise DepthEntropyError('probs must be a torch.Tensor')
    if probs.numel() == 0: raise DepthEntropyError('probs cannot be empty')
    if not torch.isfinite(probs).all(): raise DepthEntropyError('probs contains NaN/Inf')
    if (probs < -eps).any(): raise DepthEntropyError('probs contains negative entries')
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs.clamp_min(0.0) / denom
def depth_entropy(probs: torch.Tensor, *, dim: int=-1, eps: float=1e-8, normalise: bool=True) -> torch.Tensor:
    p = validate_probability_tensor(probs, dim=dim, eps=eps)
    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=dim)
    if normalise:
        n = probs.size(dim)
        ent = torch.zeros_like(ent) if n <= 1 else ent / torch.log(torch.tensor(float(n), device=probs.device, dtype=probs.dtype))
    if not torch.isfinite(ent).all(): raise DepthEntropyError('entropy contains NaN/Inf')
    return ent
