"""Small manifold operations used for consolidation and hop stability.

The implementations are intentionally conservative approximations. They are
sufficient for stable read/consolidation tests and avoid turning the subsystem
into a numerically fragile geometry experiment.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .manifold_utils import EPS, _poincare_project, wrap_angles


def log_map(geom: str, x: torch.Tensor, y: torch.Tensor, c: float | None = None) -> torch.Tensor:
    geom = geom.lower()
    if geom in {"euclid", "spatial"}:
        return y - x
    if geom == "torus":
        return wrap_angles(y - x)
    if geom == "sphere":
        x_n = F.normalize(x, dim=-1, eps=EPS)
        y_n = F.normalize(y, dim=-1, eps=EPS)
        dot = (x_n * y_n).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        direction = y_n - dot * x_n
        direction = F.normalize(direction, dim=-1, eps=EPS)
        return theta * direction
    if geom == "hyper":
        # Stable origin-chart approximation.
        return _poincare_project(y, c or 1.0) - _poincare_project(x, c or 1.0)
    return y - x


def exp_map(geom: str, x: torch.Tensor, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
    geom = geom.lower()
    if geom in {"euclid", "spatial"}:
        return x + v
    if geom == "torus":
        return wrap_angles(x + v)
    if geom == "sphere":
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        return F.normalize(torch.cos(norm_v) * F.normalize(x, dim=-1, eps=EPS) + torch.sin(norm_v) * v / norm_v, dim=-1, eps=EPS)
    if geom == "hyper":
        return _poincare_project(x + v, c or 1.0)
    return x + v


def parallel_transport(geom: str, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
    # Conservative approximation: identity transport. This is stable and avoids
    # introducing false precision in a reconstruction pack.
    return v


def frechet_mean(geom: str, X: torch.Tensor, w: torch.Tensor | None = None, c: float | None = None, iters: int = 6, lr: float = 0.5) -> torch.Tensor:
    if X.ndim != 3:
        raise ValueError("X must be [B,K,D]")
    B, K, _D = X.shape
    if w is None:
        w = torch.ones(B, K, device=X.device, dtype=X.dtype) / float(K)
    else:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(EPS)
    mu = torch.sum(w.unsqueeze(-1) * X, dim=1)
    if geom == "sphere":
        mu = F.normalize(mu, dim=-1, eps=EPS)
    elif geom == "hyper":
        mu = _poincare_project(mu, c or 1.0)
    elif geom == "torus":
        mu = wrap_angles(mu)

    for _ in range(max(1, iters)):
        v = log_map(geom, mu.unsqueeze(1).expand_as(X), X, c=c)
        v_bar = torch.sum(w.unsqueeze(-1) * v, dim=1)
        mu = exp_map(geom, mu, lr * v_bar, c=c)
    return mu
